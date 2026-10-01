"""
Step 8: collect training data with random actions.

32 sessions, 8 per scenario, 4 simulated hours each. The actions are random
(see policies.RandomActionPolicy for why) and the heater power varies between
sessions so the model can learn what the internal heat load does rather than
treating it as a constant it can ignore.

Session seeds 101-106 of each scenario become the training set and 107-108 the
held-out test set, split in Step 9. The split is decided here, at collection
time, so there is no way to peek at the test sessions while choosing features.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from windwindow import config as cfg
from windwindow.policies import FixedActionPolicy, RandomActionPolicy
from windwindow.simulator import run_session, summarise
from windwindow.weather import SCENARIOS, make_scenario

OUTDIR = pathlib.Path("data/train")
SEEDS = list(range(101, 109))              # 8 sessions per scenario
TEST_SEEDS = {107, 108}                    # held out for the Step 9 split
# Fixed actions for the excitation sessions, and which of them are held out.
EXCITE_ACTIONS = [(1.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.0, 0.0), (0.75, 1.0), (0.25, 0.0)]
EXCITE_TEST = {4, 5}
HEATERS = [17.6, 8.8, 17.6, 13.2, 17.6, 0.0, 17.6, 11.0]
DURATION_S = 4 * 3600.0

# How far each session's starting state is pushed away from the scenario
# default. Wide on purpose: the room has to be started well below the outdoor
# temperature sometimes, or the log never contains a single row of "the window
# was open while it was hotter outside" -- and that is the case the model is
# most likely to get wrong and least likely to see from a sensible policy.
START_SPREAD_IN = (-6.0, 4.0)
START_SPREAD_WALL = (-6.0, 4.0)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("*.csv"):
        old.unlink()

    rows = []
    for si, name in enumerate(SCENARIOS):
        for i, seed in enumerate(SEEDS):
            # One RNG per session, seeded from the scenario as well as the
            # session number -- otherwise every scenario gets the identical
            # random action sequence and the coverage is a quarter of what the
            # session count suggests.
            rng = np.random.default_rng(900_000 + si * 1_000 + seed)
            base = make_scenario(name, seed=seed, duration_s=DURATION_S,
                                 heater_w=HEATERS[i])
            sc = make_scenario(
                name, seed=seed, duration_s=DURATION_S, heater_w=HEATERS[i],
                t_in0=base.t_in0 + rng.uniform(*START_SPREAD_IN),
                t_wall0=base.t_wall0 + rng.uniform(*START_SPREAD_WALL),
            )
            pol = RandomActionPolicy(seed=int(rng.integers(0, 2**31 - 1)))
            sid = f"{name}_s{seed}"
            df = run_session(sc, pol, sensor_seed=50_000 + si * 977 + seed * 7 + i,
                             session_id=sid)
            df["split"] = "test" if seed in TEST_SEEDS else "train"
            df.to_csv(OUTDIR / f"{sid}.csv", index=False)
            s = summarise(df)
            s["heater_w"] = HEATERS[i]
            s["t_in0"] = sc.t_in0
            s["t_wall0"] = sc.t_wall0
            s["split"] = df["split"].iloc[0]
            s["distinct_actions"] = len(df.groupby(["u", "f"]))
            rows.append(s)

    # ---------------------------------------------------------------------
    # Excitation sessions: one fixed action for four hours, started a long way
    # from equilibrium. See policies.FixedActionPolicy for why these are here.
    # ---------------------------------------------------------------------
    for si, name in enumerate(SCENARIOS):
        for j, (u, f) in enumerate(EXCITE_ACTIONS):
            rng = np.random.default_rng(700_000 + si * 1_000 + j)
            seed = 300 + j
            base = make_scenario(name, seed=seed, duration_s=DURATION_S)
            # Start relative to the outdoor temperature, often well below it,
            # so the window-open-while-hotter-outside case lasts a while.
            t_out0 = float(base.t_out_grid[0])
            t_in0 = float(np.clip(t_out0 + rng.uniform(-8.0, 6.0), 14.0, 38.0))
            t_wall0 = float(np.clip(t_in0 + rng.uniform(-3.0, 6.0), 14.0, 40.0))
            heater = float(rng.choice([0.0, 8.8, 17.6]))
            sc = make_scenario(name, seed=seed, duration_s=DURATION_S,
                               heater_w=heater, t_in0=t_in0, t_wall0=t_wall0)
            sid = f"{name}_x{seed}_u{u:g}f{f:g}"
            df = run_session(sc, FixedActionPolicy(u, f),
                             sensor_seed=70_000 + si * 977 + j, session_id=sid)
            df["split"] = "test" if j in EXCITE_TEST else "train"
            df.to_csv(OUTDIR / f"{sid}.csv", index=False)
            rec = summarise(df)
            rec["heater_w"] = heater
            rec["t_in0"] = t_in0
            rec["t_wall0"] = t_wall0
            rec["split"] = df["split"].iloc[0]
            rec["distinct_actions"] = 1
            rows.append(rec)

    s = pd.DataFrame(rows)
    files = sorted(OUTDIR.glob("*.csv"))
    n_train = int((s.split == "train").sum())
    n_test = int((s.split == "test").sum())

    print(f"sessions written: {len(files)}  ({n_train} train, {n_test} held-out test)")
    print(f"rows per session: {int(DURATION_S / cfg.DT_CONTROL)}   "
          f"total rows: {len(files) * int(DURATION_S / cfg.DT_CONTROL)}")
    print(f"  {int((s.policy=='random').sum())} random-action sessions, "
          f"{int((s.policy=='fixed').sum())} fixed-action excitation sessions")
    print(f"\nby scenario:")
    print(s.groupby("scenario").agg(
        sessions=("session_id", "count"),
        start_T_in=("t_in0", "mean"),
        mean_T_in=("mean_t_in", "mean"),
        max_T_in=("max_t_in", "max"),
        fan_min=("fan_minutes", "mean"),
        open_min=("window_open_minutes", "mean"),
        moves=("moves", "mean"),
    ).round(2).to_string())

    # How much of the action space did the random policy actually visit?
    allrows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    grid = allrows.groupby(["u", "f"]).size().rename("rows").reset_index()
    print(f"\ndistinct (u, f) action combinations visited: {len(grid)}")
    print(grid.to_string(index=False))

    print(f"\nstate coverage across all {len(allrows)} rows:")
    for col, lbl in (("t_in_true", "T_in"), ("t_wall_true", "T_wall"),
                     ("t_out_true", "T_out")):
        print(f"  {lbl:7s} {allrows[col].min():6.2f} .. {allrows[col].max():6.2f}")
    d = allrows.t_out_true - allrows.t_in_true
    w = allrows.t_wall_true - allrows.t_in_true
    print(f"  T_out-T_in {d.min():+6.2f} .. {d.max():+6.2f}")
    print(f"  T_wall-T_in {w.min():+6.2f} .. {w.max():+6.2f}")
    # Did we see the window open while it was hotter outside? That is the case
    # a sensible policy would never generate, and the one the model most needs.
    bad_idea = ((allrows.u > 0.5) & (d > 0.5)).sum()
    print(f"\nrows with the window open while it was warmer outside: {bad_idea} "
          f"({bad_idea/len(allrows)*100:.1f}%)  <- the cases a sensible policy never shows you")

    s.to_csv("results/tables/step08_training_sessions.csv", index=False)
    print("  wrote results/tables/step08_training_sessions.csv")

    checks = [
        ("at least 15 sessions collected (PLAN asks for 15+)", len(files) >= 15),
        ("sessions split into train and test", n_train > 0 and n_test > 0),
        ("both window and fan were varied", len(grid) >= 8),
        ("enough rows with the window open while warmer outside", bad_idea > 600),
        ("heater power varies across sessions", s.heater_w.nunique() >= 3),
        ("all four scenarios represented", s.scenario.nunique() == 4),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 8 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
