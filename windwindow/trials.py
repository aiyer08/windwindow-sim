"""
Goal G2: paired open/closed trials.

A "paired trial" freezes the room in one state and then asks what would have
happened under each action, from that identical starting point. Because this
is a simulation we can genuinely do that, which a hardware rig cannot: you
only get to live one of the two futures.

For each trial:

  ground truth   step the real physics forward ten minutes twice from the same
                 true state, once with the window shut and once with it open
                 (taking the better of fan-on and fan-off), using the real
                 future outdoor temperature. Whichever leaves the room cooler
                 is the right answer.
  the model      sees only the sensor readings and the 5-minute trends, and
                 says open if its predicted benefit is positive.
  the baseline rule sees the same readings and says open if it is cooler outside.

Trials where the two futures land within DEAD_BAND of each other are marked as
ties: either answer is counted correct, because the choice genuinely does not
matter there and scoring it either way would be noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .policies import RandomActionPolicy, SimpleRulePolicy
from .room import Room
from .simulator import run_session
from .weather import SCENARIOS, make_scenario

DEAD_BAND = 0.05        # degC; closer than this and the choice does not matter
OPEN_ACTIONS = ((1.0, 0.0), (1.0, 1.0))


def _roll_true(t_in, t_wall, u, f, scenario, t0, heater_w):
    """Step the real physics ten minutes with the real future weather."""
    room = Room(t_in=t_in, t_wall=t_wall, heater_w=heater_w)
    for k in range(int(cfg.HORIZON_S / cfg.DT_PHYSICS)):
        room.step(u, f, scenario.t_out(t0 + k * cfg.DT_PHYSICS))
    return room.t_in


def build_trial_pool(seeds=(21, 22, 23), duration_s=3 * 3600.0) -> pd.DataFrame:
    """Random-action sessions on fresh weather, as the pool to draw trials from.

    Fresh seeds, so the weather is new. Random actions, so the states are not
    the ones any particular policy happens to visit -- otherwise the trial set
    would be biased toward whichever controller generated it.
    """
    from .features import featurise_session

    frames = []
    for si, name in enumerate(SCENARIOS):
        for seed in seeds:
            sc = make_scenario(name, seed=seed, duration_s=duration_s)
            pol = RandomActionPolicy(seed=400_000 + si * 100 + seed)
            df = run_session(sc, pol, sensor_seed=80_000 + si * 31 + seed,
                             session_id=f"trial_{name}_s{seed}")
            frames.append(featurise_session(df))
    pool = pd.concat(frames, ignore_index=True)
    # Need 5 minutes of history behind the row for the trends, and 10 minutes
    # of session left ahead of it for the outcome.
    win = int(cfg.SLOPE_WINDOW_S / cfg.DT_CONTROL)
    keep = []
    for _, g in pool.groupby("session_id", sort=False):
        g = g.reset_index(drop=True)
        last_ok = len(g) - int(cfg.HORIZON_S / cfg.DT_CONTROL) - 1
        keep.append(g.iloc[win:last_ok])
    return pd.concat(keep, ignore_index=True)


def run_trials(pool: pd.DataFrame, predictor, n: int = 30, seed: int = 0,
               duration_s: float = 3 * 3600.0) -> pd.DataFrame:
    """Draw n trials, stratified across the four scenarios, and score them."""
    rng = np.random.default_rng(seed)
    per = max(1, n // len(SCENARIOS))
    picks = []
    for name in SCENARIOS:
        sub = pool[pool.scenario == name]
        take = min(per, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    chosen = pd.concat(picks)
    if len(chosen) < n:                      # top up if n is not divisible by 4
        rest = pool.drop(chosen.index)
        chosen = pd.concat([chosen,
                            rest.iloc[rng.choice(len(rest), size=n - len(chosen),
                                                 replace=False)]])
    chosen = chosen.iloc[:n]

    scen_cache = {}
    rows = []
    for _, r in chosen.iterrows():
        key = (r["scenario"], int(r["seed"]))
        if key not in scen_cache:
            scen_cache[key] = make_scenario(r["scenario"], seed=int(r["seed"]),
                                            duration_s=duration_s)
        sc = scen_cache[key]
        t0 = float(r["t_s"])
        ti, tw = float(r["t_in_true"]), float(r["t_wall_true"])
        heater = float(r["heater_w"])

        shut = _roll_true(ti, tw, 0.0, 0.0, sc, t0, heater)
        opened = min(_roll_true(ti, tw, u, f, sc, t0, heater) for u, f in OPEN_ACTIONS)
        true_benefit = shut - opened                 # positive: opening is better
        tie = abs(true_benefit) <= DEAD_BAND
        truth_open = true_benefit > 0

        state = {"t_in": r["t_in"], "t_wall": r["t_wall"], "t_out": r["t_out"],
                 "heater_w": heater, "slope_in": r["slope_in"],
                 "slope_out": r["slope_out"]}
        p_shut = predictor(state, 0.0, 0.0)
        p_open = min(predictor(state, u, f) for u, f in OPEN_ACTIONS)
        model_benefit = p_shut - p_open
        model_open = model_benefit > 0

        rule_open = bool(r["t_out"] < r["t_in"] - SimpleRulePolicy.OPEN_GAP)

        rows.append({
            "scenario": r["scenario"], "seed": int(r["seed"]), "t_s": t0,
            "t_in_true": ti, "t_wall_true": tw, "t_out_true": r["t_out_true"],
            "u_at_row": r["u"], "f_at_row": r["f"],
            "true_benefit": true_benefit, "tie": tie, "truth_open": truth_open,
            "model_benefit": model_benefit, "model_open": model_open,
            "rule_open": rule_open,
            "model_correct": bool(tie or (model_open == truth_open)),
            "rule_correct": bool(tie or (rule_open == truth_open)),
        })
    return pd.DataFrame(rows)


def mcnemar(a_correct, b_correct):
    """Exact McNemar test on two paired sets of right/wrong answers.

    Only the trials where the two disagree carry information. If A is right and
    B wrong n_a times, and the reverse n_b times, then under "they are equally
    good" n_a is a fair coin flip out of n_a + n_b.
    """
    from scipy.stats import binomtest

    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n_a = int((a & ~b).sum())
    n_b = int((~a & b).sum())
    if n_a + n_b == 0:
        return n_a, n_b, 1.0
    p = binomtest(n_a, n_a + n_b, 0.5).pvalue
    return n_a, n_b, float(p)
