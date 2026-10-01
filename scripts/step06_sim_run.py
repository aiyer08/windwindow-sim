"""Step 6: does one 3-hour run produce a CSV?"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from windwindow import config as cfg, plotting as P
from windwindow.simulator import run_session, summarise, LOG_COLUMNS
from windwindow.policies import TimerPolicy
from windwindow.weather import make_scenario

OUT = pathlib.Path("data/step06_demo_timer.csv")

def main():
    sc = make_scenario("hot_then_cooling", seed=1, duration_s=3 * 3600.0)
    df = run_session(sc, TimerPolicy())
    df.to_csv(OUT, index=False)

    checks = []
    expected_rows = int(3 * 3600 / cfg.DT_CONTROL)
    print(f"rows: {len(df)} (expected {expected_rows} for a 3-hour run at "
          f"{cfg.DT_CONTROL:.0f} s per row)")
    print(f"columns: {len(df.columns)}")
    print(f"file: {OUT}  ({OUT.stat().st_size/1024:.1f} kB)")
    checks.append(("3 hours at 30 s per row", len(df) == expected_rows))
    checks.append(("all expected columns present", list(df.columns) == LOG_COLUMNS))
    checks.append(("no missing values", not df.isna().any().any()))
    checks.append(("CSV written and non-empty", OUT.exists() and OUT.stat().st_size > 1000))
    checks.append(("timestamps evenly spaced",
                   bool(np.allclose(np.diff(df["t_s"]), cfg.DT_CONTROL))))
    # The policy must only ever see measured values, so truth and measurement
    # must actually differ.
    checks.append(("measured differs from truth",
                   float((df["t_in"] - df["t_in_true"]).abs().mean()) > 0.01))
    # Energy bookkeeping must add up.
    fan_wh_expected = df["fan_on_s"].sum() / 3600.0 * cfg.FAN_ELEC_W
    checks.append(("fan energy matches fan runtime",
                   abs(df["fan_wh"].sum() - fan_wh_expected) < 1e-6))

    s = summarise(df)
    print("\nsession summary")
    for k, v in s.items():
        print(f"  {k:24s} {v if isinstance(v, str) else round(float(v), 3)}")

    # ------------------------------------------------------------------ figure
    P.plot_session(
        df, "Step 6  ·  One 3-hour run of the timer policy",
        note=f"The timer opens the window and runs the fan for the first 15 min of every "
             f"30, whatever the weather is doing.    Fan ran {s['fan_minutes']:.0f} of "
             f"{s['minutes']:.0f} min ({s['fan_wh']:.2f} Wh); "
             f"{s['degree_minutes_over_26']:.0f} degree-minutes over 26 °C.",
        path="results/figures/step06_timer_run.png",
    )

    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 6 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
