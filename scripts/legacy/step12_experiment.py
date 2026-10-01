"""
Step 12: the full experiment.

3 policies x 4 scenarios x 3 repeats = 36 runs.

The repeats use weather seeds 11, 12 and 13, which appear nowhere in the
training data (seeds 101-108 and 300-305). So the model is being asked about
weather it has never seen, on top of being scored on sessions it never trained
on.

Within one (scenario, repeat) cell all three policies get the identical
weather trace, the identical starting state and the identical sensor
calibration errors. The runs are paired, so any difference between them is the
policy and nothing else.
"""

import json
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from windwindow.policies import ModelPolicy, SimpleRulePolicy, TimerPolicy
from windwindow.simulator import run_session, summarise
from windwindow.weather import SCENARIOS, make_scenario

OUTDIR = pathlib.Path("data/experiments")
REPEAT_SEEDS = (11, 12, 13)
DURATION_S = 3 * 3600.0

def main():
    with open("results/model.pkl", "rb") as fh:
        blob = pickle.load(fh)
    model, best_name = blob[blob["best"]], blob["best"]
    # Use the fan margin Step 11 settled on, rather than assuming the default.
    # They happen to be the same value, but reading it keeps the dependency
    # explicit so a different choice in Step 11 would actually take effect here.
    choice = json.loads(pathlib.Path("results/fan_margin.json").read_text())
    fan_margin = float(choice["fan_margin"])
    print(f"model policy uses: {best_name}, fan margin "
          f"{fan_margin:.2f} °C (from Step 11)\n")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("*.csv"):
        old.unlink()

    rows = []
    for name in SCENARIOS:
        for seed in REPEAT_SEEDS:
            sc = make_scenario(name, seed=seed, duration_s=DURATION_S)
            # Same sensor seed for all three, so they inherit the same
            # calibration errors and the comparison stays paired.
            sensor_seed = 20_000 + seed * 13
            for pol in (TimerPolicy(), SimpleRulePolicy(),
                        ModelPolicy(model.as_predictor(),
                                    fan_margin=fan_margin)):
                sid = f"{name}_s{seed}_{pol.name}"
                df = run_session(sc, pol, sensor_seed=sensor_seed, session_id=sid)
                df.to_csv(OUTDIR / f"{sid}.csv", index=False)
                rows.append(summarise(df))

    s = pd.DataFrame(rows)
    s.to_csv("results/tables/step12_runs.csv", index=False)

    n_files = len(list(OUTDIR.glob("*.csv")))
    print(f"runs logged: {len(s)}  ({s.policy.nunique()} policies x "
          f"{s.scenario.nunique()} scenarios x {len(REPEAT_SEEDS)} repeats)")
    print(f"CSVs written: {n_files}\n")

    agg = (s.groupby("policy")
            .agg(degree_min_over_26=("degree_minutes_over_26", "mean"),
                 minutes_over_26=("minutes_over_26", "mean"),
                 minutes_under_24=("minutes_under_24", "mean"),
                 mean_T_in=("mean_t_in", "mean"),
                 max_T_in=("max_t_in", "max"),
                 fan_minutes=("fan_minutes", "mean"),
                 fan_wh=("fan_wh", "mean"),
                 open_minutes=("window_open_minutes", "mean"),
                 moves=("moves", "mean"))
            .reindex(["timer", "simple_rule", "model"]))
    print("averaged over all 12 runs per policy:")
    print(agg.round(2).to_string())

    print("\nper scenario (mean of 3 repeats):")
    for metric, label in (("degree_minutes_over_26", "degree-minutes over 26 °C"),
                          ("fan_minutes", "fan minutes")):
        piv = s.pivot_table(index="scenario", columns="policy", values=metric)
        piv = piv.reindex(columns=["timer", "simple_rule", "model"])
        print(f"\n  {label}")
        print("    " + piv.round(1).to_string().replace("\n", "\n    "))

    s.to_csv("results/tables/step12_runs.csv", index=False)
    print("\n  wrote results/tables/step12_runs.csv")

    # Verify the pairing directly: within each (scenario, repeat) cell the three
    # policies must have seen the identical outdoor trace, and must have taken
    # different actions. Comparing outcomes would not prove it -- in the cool
    # scenario all three correctly score zero degree-minutes.
    paired_weather, policies_differ = True, True
    distinct, model_vs_rule_differs = [], 0
    for name in SCENARIOS:
        for seed in REPEAT_SEEDS:
            frames = {pol: pd.read_csv(OUTDIR / f"{name}_s{seed}_{pol}.csv")
                      for pol in ("timer", "simple_rule", "model")}
            ref = frames["timer"]["t_out_true"].to_numpy()
            for pol, d in frames.items():
                if not np.allclose(d["t_out_true"].to_numpy(), ref):
                    paired_weather = False
            acts = {pol: tuple(d["u"].round(3)) for pol, d in frames.items()}
            distinct.append(len(set(acts.values())))
            if acts["model"] != acts["simple_rule"]:
                model_vs_rule_differs += 1
            if acts["timer"] in (acts["model"], acts["simple_rule"]):
                policies_differ = False
    print(f"\npairing check: identical weather across policies in every cell: "
          f"{paired_weather}")
    print(f"               the timer never matched a sensing policy: {policies_differ}")
    print(f"               model and baseline rule chose differently in "
          f"{model_vs_rule_differs} of {len(distinct)} cells")
    print(f"               (they agree in the cool scenario, where both correctly")
    print(f"                keep the window shut almost the whole run)")

    checks = [
        ("36 runs logged", len(s) == 36),
        ("36 CSVs written", n_files == 36),
        ("3 policies present", s.policy.nunique() == 3),
        ("4 scenarios present", s.scenario.nunique() == 4),
        ("3 repeats per cell",
         bool((s.groupby(["scenario", "policy"]).size() == 3).all())),
        ("every run is a full 180 minutes", bool((s.minutes == 180).all())),
        ("runs are truly paired: same weather for all three policies",
         paired_weather),
        ("the timer never coincided with a sensing policy", policies_differ),
        ("the model and the baseline rule disagree in most cells",
         model_vs_rule_differs > len(distinct) / 2),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 12 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
