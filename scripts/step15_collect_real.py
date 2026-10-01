"""
Step 15: collect identification data on real weather.

The synthetic three-hour sessions in step 8 were adequate for fitting a
ten-minute predictor, but they cover neither the temperature range of a real
heat wave (Phoenix reaches 45.8 degC) nor the time span a planner reasons
over. This step logs 48-hour sessions across all five climates.

Actions are randomised or held constant, for the reason given in step 8: a
sensible policy never demonstrates what happens when you ventilate at the
wrong moment, and the model has to know that. Two sessions per location are
kept for validation and are never used for fitting.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from windwindow import config as cfg
from windwindow.policies import FixedActionPolicy, RandomActionPolicy
from windwindow.realweather import LOCATIONS, build
from windwindow.simulator import run_session, summarise

OUTDIR = pathlib.Path("data/identify")
HOURS = 48.0
# Two randomised sessions and two held-action sessions per location. The held
# actions are partial openings, which the old data barely contained because
# the controller only ever chose 0 or 1.
RANDOM_SEEDS = [1, 2, 3]
FIXED_ACTIONS = [(0.35, 0.0), (0.6, 1.0), (1.0, 0.45)]
VALIDATION_SEEDS = {3}          # held out from fitting
VALIDATION_ACTIONS = {(1.0, 0.45)}

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("*.csv"):
        old.unlink()

    rows = []
    for li, loc in enumerate(LOCATIONS):
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(500_000 + li * 1_000 + seed)
            w = build(loc, seed=seed, hours=HOURS,
                      start_hour=int(rng.integers(0, 24)),
                      heater_w=float(rng.choice([0.0, 8.8, 17.6])),
                      t_in0=None, t_wall0=None)
            # Push the starting state around so the transients are informative.
            w.t_in0 += float(rng.uniform(-3, 5))
            w.t_wall0 += float(rng.uniform(-3, 5))
            pol = RandomActionPolicy(seed=int(rng.integers(0, 2**31 - 1)),
                                     hold_min=(8.0, 45.0))
            sid = f"{loc}_rand{seed}"
            df = run_session(w, pol, sensor_seed=90_000 + li * 97 + seed,
                             session_id=sid)
            df["split"] = "validate" if seed in VALIDATION_SEEDS else "fit"
            df["location"] = loc
            df.to_csv(OUTDIR / f"{sid}.csv", index=False)
            rec = summarise(df)
            rec.update(location=loc, kind="random", heater_w=w.heater_w,
                       split=df["split"].iloc[0])
            rows.append(rec)

        for ai, (u, f) in enumerate(FIXED_ACTIONS):
            rng = np.random.default_rng(600_000 + li * 1_000 + ai)
            w = build(loc, seed=10 + ai, hours=HOURS,
                      start_hour=int(rng.integers(0, 24)),
                      heater_w=float(rng.choice([0.0, 17.6])))
            w.t_in0 += float(rng.uniform(-4, 6))
            w.t_wall0 += float(rng.uniform(-4, 6))
            sid = f"{loc}_hold_u{u:g}f{f:g}"
            df = run_session(w, FixedActionPolicy(u, f),
                             sensor_seed=95_000 + li * 97 + ai, session_id=sid)
            df["split"] = "validate" if (u, f) in VALIDATION_ACTIONS else "fit"
            df["location"] = loc
            df.to_csv(OUTDIR / f"{sid}.csv", index=False)
            rec = summarise(df)
            rec.update(location=loc, kind=f"hold u={u:g} f={f:g}",
                       heater_w=w.heater_w, split=df["split"].iloc[0])
            rows.append(rec)

    s = pd.DataFrame(rows)
    files = sorted(OUTDIR.glob("*.csv"))
    n_fit = int((s.split == "fit").sum())
    n_val = int((s.split == "validate").sum())
    rows_each = int(HOURS * 3600 / cfg.DT_CONTROL)

    print(f"sessions: {len(files)}  ({n_fit} for fitting, {n_val} held out)")
    print(f"{HOURS:.0f} h each at {cfg.DT_CONTROL:.0f} s per row = "
          f"{rows_each} rows per session, {len(files) * rows_each:,} total")
    print()
    print(s.groupby("location").agg(
        sessions=("session_id", "count"),
        T_in_min=("mean_t_in", "min"), T_in_max=("max_t_in", "max"),
        fan_h=("fan_minutes", lambda v: round(v.mean() / 60, 1)),
        open_h=("window_open_minutes", lambda v: round(v.mean() / 60, 1)),
    ).round(2).to_string())

    allrows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"\nstate coverage across {len(allrows):,} rows:")
    for col, lbl in (("t_in_true", "T_in"), ("t_wall_true", "T_wall"),
                     ("t_out_true", "T_out")):
        print(f"  {lbl:7s} {allrows[col].min():6.2f} .. {allrows[col].max():6.2f} degC")
    d = allrows.t_out_true - allrows.t_in_true
    print(f"  T_out - T_in {d.min():+6.2f} .. {d.max():+6.2f} degC")
    grid = allrows.groupby([allrows.u.round(2), allrows.f.round(2)]).size()
    print(f"\ndistinct (u, f) combinations: {len(grid)}")
    frac = float(((allrows.u > 0.01) & (allrows.u < 0.99)).mean())
    print(f"rows at a partial opening (0 < u < 1): "
          f"{int(((allrows.u > 0.01) & (allrows.u < 0.99)).sum()):,} "
          f"({frac*100:.0f}% of rows)")

    s.to_csv("results/tables/step15_identification_sessions.csv", index=False)
    print("\n  wrote results/tables/step15_identification_sessions.csv")

    checks = [
        ("sessions written for all five climates", s.location.nunique() == 5),
        ("both fitting and validation sessions exist", n_fit > 0 and n_val > 0),
        ("48 h per session", bool((s.minutes == HOURS * 60).all())),
        ("partial openings are a third or more of rows",
         float(((allrows.u > 0.01) & (allrows.u < 0.99)).mean()) > 0.33),
        ("covers the real heat-wave range", allrows.t_out_true.max() > 43),
        ("covers cool nights", allrows.t_out_true.min() < 16),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(c[1] for c in checks)
    print(f"\nSTEP 15 DONE-WHEN: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1

if __name__ == "__main__":
    raise SystemExit(main())
