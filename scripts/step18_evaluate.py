"""
Step 18: what the controller achieves, by climate.

Five real climates, two weather seeds each, 48 hours per run. The fan penalty
was fixed in step 17 on Palo Alto seed 1 and is not touched here, so every
number below comes from weather and seeds the controller was not tuned on.

Each cell also runs a reference controller given perfect foresight and the
true room parameters. It is subject to the same block structure and the same
re-planning interval, so the gap between it and the real controller measures
what better information would buy, not what a different algorithm would.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from windwindow import config as cfg
from windwindow.forecast import Forecast
from windwindow.mpc import MPCPolicy
from windwindow.policies import FixedActionPolicy, SimpleRulePolicy, TimerPolicy
from windwindow.realweather import LOCATIONS, build
from windwindow.simulator import run_session, summarise
from windwindow.statespace import StateSpaceModel, true_params

SEEDS = (2, 3)
HOURS = 48.0
OUTDIR = pathlib.Path("data/evaluate")

def main():
    blob = json.load(open("results/statespace.json"))
    model = StateSpaceModel(theta=np.array(blob["theta"]))
    fan_penalty = json.load(open("results/fan_penalty.json"))["fan_penalty"]
    oracle_model = StateSpaceModel(theta=true_params())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("*.csv"):
        old.unlink()

    print(f"fan penalty {fan_penalty:g}, fixed in step 17 on Palo Alto seed 1")
    print(f"{len(LOCATIONS)} climates x {len(SEEDS)} seeds x {HOURS:.0f} h\n")

    rows = []
    t_start = time.time()
    for loc in LOCATIONS:
        for seed in SEEDS:
            w = build(loc, seed=seed, hours=HOURS, start_hour=0)
            fc = Forecast.build(w.t_out_grid, seed=seed)
            perfect = Forecast.build(w.t_out_grid, seed=seed, perfect=True)
            ss = 30_000 + seed * 17
            policies = [
                ("sealed", FixedActionPolicy(0.0, 0.0)),
                ("open throughout", FixedActionPolicy(1.0, 0.0)),
                ("timer", TimerPolicy()),
                ("threshold rule", SimpleRulePolicy()),
                ("controller", MPCPolicy(model, fc, fan_penalty=fan_penalty)),
                ("reference, perfect information",
                 MPCPolicy(oracle_model, perfect, fan_penalty=fan_penalty)),
            ]
            for label, pol in policies:
                df = run_session(w, pol, sensor_seed=ss,
                                 session_id=f"{loc}_s{seed}_{label.replace(' ','_')}")
                df.to_csv(OUTDIR / f"{loc}_s{seed}_{label.replace(' ', '_')}.csv",
                          index=False)
                s = summarise(df)
                s.update(location=loc, label=w.label, policy=label, seed=seed,
                         t_out_min=float(w.t_out_grid.min()),
                         t_out_max=float(w.t_out_grid.max()),
                         t_out_mean=float(w.t_out_grid.mean()),
                         night_mean=w.summary()["night_mean"],
                         hours_over_26=float(s["minutes_over_26"]) / 60.0)
                rows.append(s)
            print(f"  {loc:11s} seed {seed}  done  "
                  f"({time.time()-t_start:5.0f} s elapsed)")

    d = pd.DataFrame(rows)
    d.to_csv("results/tables/step18_runs.csv", index=False)

    # ---------------------------------------------------------------- report
    piv = d.pivot_table(index="location", columns="policy",
                        values="degree_minutes_over_26")
    order = ["sealed", "open throughout", "timer", "threshold rule",
             "controller", "reference, perfect information"]
    piv = piv.reindex(columns=order)
    print(f"\ndegree-minutes above {cfg.COMFORT_C:.0f} degC, mean of "
          f"{len(SEEDS)} seeds:")
    print(piv.round(0).to_string())

    print(f"\nwhat the controller achieves, by climate:")
    print(f"  {'climate':12s} {'night':>6s} {'vs sealed':>10s} {'vs rule':>9s} "
          f"{'gap to ref':>11s} {'hours>26':>9s} {'fan kWh':>8s}")
    summary = []
    for loc in LOCATIONS:
        sub = d[d.location == loc]
        g = sub.groupby("policy")
        sealed = g.degree_minutes_over_26.mean()["sealed"]
        rule = g.degree_minutes_over_26.mean()["threshold rule"]
        ctl = g.degree_minutes_over_26.mean()["controller"]
        ref = g.degree_minutes_over_26.mean()["reference, perfect information"]
        hrs = g.hours_over_26.mean()["controller"]
        fan = g.fan_wh.mean()["controller"] / 1000.0
        night = sub.night_mean.iloc[0]
        vs_sealed = (1 - ctl / sealed) * 100 if sealed > 0 else float("nan")
        vs_rule = (1 - ctl / rule) * 100 if rule > 0 else float("nan")
        gap = (ctl / ref - 1) * 100 if ref > 0 else float("nan")
        print(f"  {loc:12s} {night:6.1f} {vs_sealed:9.1f}% {vs_rule:8.1f}% "
              f"{gap:10.1f}% {hrs:9.1f} {fan:8.2f}")
        summary.append({"location": loc, "label": sub.label.iloc[0],
                        "night_mean_c": night, "sealed": sealed, "rule": rule,
                        "controller": ctl, "reference": ref,
                        "vs_sealed_pct": vs_sealed, "vs_rule_pct": vs_rule,
                        "gap_to_reference_pct": gap,
                        "hours_over_26": hrs, "fan_kwh": fan,
                        "t_out_max": sub.t_out_max.iloc[0]})
    s = pd.DataFrame(summary)
    s.to_csv("results/tables/step18_by_climate.csv", index=False)
    print("\n  wrote results/tables/step18_runs.csv")
    print("  wrote results/tables/step18_by_climate.csv")

    print(f"\n  the single number that decides whether this is worth "
          f"installing is the")
    print(f"  overnight low. Where it falls well below the comfort threshold "
          f"there is cool")
    print(f"  air to store; where it does not, no control strategy can "
          f"manufacture any.")
    corr = s[["night_mean_c", "vs_sealed_pct"]].corr().iloc[0, 1]
    print(f"  correlation between overnight mean and improvement on a sealed "
          f"room: {corr:+.2f}")

    checks = [
        ("every climate and seed ran",
         len(d) == len(LOCATIONS) * len(SEEDS) * 6),
        ("the controller beats a sealed room everywhere",
         bool((s.vs_sealed_pct > 0).all())),
        ("the controller is within 25% of the perfect-information reference",
         bool((s.gap_to_reference_pct < 25).all())),
        ("results vary by climate, so the recommendation is climate specific",
         float(s.vs_sealed_pct.std()) > 5),
        ("fan energy stays modest", bool((s.fan_kwh < 3.0).all())),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(c[1] for c in checks)
    print(f"\nSTEP 18 DONE-WHEN: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1

if __name__ == "__main__":
    raise SystemExit(main())
