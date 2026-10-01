"""
Step 16: identify the room dynamics from the logged data.

Fits the twelve coefficients of the two-state model in statespace.py, using
only the fitting sessions, then measures prediction error on the held-out
sessions at horizons from ten minutes to six hours.

The comparison that matters is against the ten-minute predictor from step 10.
That model is accurate at ten minutes and has nothing to say beyond it. A
planner needs to know where the room will be in four hours.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.statespace import (PARAM_NAMES, StateSpaceModel, true_params)

DATADIR = pathlib.Path("data/identify")
MODEL_PATH = pathlib.Path("results/statespace.json")
HORIZONS = (10, 30, 60, 180, 360)

def load(split=None):
    out = []
    for f in sorted(DATADIR.glob("*.csv")):
        df = pd.read_csv(f)
        if split is None or df["split"].iloc[0] == split:
            out.append(df)
    if not out:
        raise FileNotFoundError("no identification sessions; run step15 first")
    return out

def main():
    fit_sessions = load("fit")
    val_sessions = load("validate")
    print(f"fitting on {len(fit_sessions)} sessions, validating on "
          f"{len(val_sessions)} held-out sessions\n")

    model = StateSpaceModel()
    t0 = time.time()
    model.fit(fit_sessions, horizons_min=(60.0, 360.0), stride_min=45.0)
    print(f"    converged in {time.time()-t0:.1f} s "
          f"({model.fit_info['n_fev']} evaluations)\n")

    # --- how close did identification get, without being told? ------------
    tp = true_params()
    ident = model.theta
    units = ["J/K", "J/K", "W/K", "W/K", "W/K", "W/K", "W/K", "W/K", "W/K", "W"]
    print("identified parameters against the values implied by config.py")
    print("(the true values are shown for comparison and took no part in the fit)\n")
    print(f"  {'parameter':11s} {'identified':>12s} {'true':>12s} {'error':>9s}  unit")
    rows = []
    for n, a, b, un in zip(PARAM_NAMES, ident, tp, units):
        rel = (a - b) / abs(b) * 100 if b != 0 else float("nan")
        fmt = "{:12.0f}" if un == "J/K" else "{:12.2f}"
        print(f"  {n:11s} " + fmt.format(a) + " " + fmt.format(b)
              + f" {rel:+8.1f}%  {un}")
        rows.append({"parameter": n, "unit": un, "identified": a,
                     "implied_by_config": b, "relative_error_pct": rel})
    pd.DataFrame(rows).to_csv("results/tables/step16_coefficients.csv", index=False)

    err_pct = np.abs([r["relative_error_pct"] for r in rows])
    c_in_hat, c_wall_hat = ident[0], ident[1]
    ua_hat = 1.0 / (1.0 / ident[2] + 1.0 / ident[8]) + ident[5]
    ua_true = 1.0 / (1.0 / cfg.K_IN_WALL_BASE + 1.0 / cfg.K_WALL_OUT) + cfg.K_LEAK
    tau_hat = c_wall_hat / (ident[2] + ident[8]) / 3600.0
    tau_true = cfg.C_WALL / (cfg.K_IN_WALL_BASE + cfg.K_WALL_OUT) / 3600.0
    print(f"\n  median parameter error {np.median(err_pct):.0f}%, "
          f"worst {err_pct.max():.0f}% ({PARAM_NAMES[int(np.argmax(err_pct))]}, "
          f"the smallest term in the model)")
    print(f"  heat loss coefficient, window shut: {ua_hat:.1f} W/K identified "
          f"against {ua_true:.1f} true ({(ua_hat/ua_true-1)*100:+.1f}%)")
    print(f"  structural time constant: {tau_hat:.1f} h identified against "
          f"{tau_true:.1f} h true ({(tau_hat/tau_true-1)*100:+.1f}%)")
    print(f"  all parameters positive and physically readable: "
          f"{bool((ident > 0).all())}")

    # --- prediction error by horizon --------------------------------------
    print(f"\nprediction error on the {len(val_sessions)} held-out sessions:")
    print(f"  {'horizon':>9s} {'n':>7s} {'MAE':>8s} {'RMSE':>8s} {'bias':>8s}")
    err_rows = model.multistep_error(val_sessions, horizons_min=HORIZONS)
    for r in err_rows:
        print(f"  {r['horizon_min']:6d} min {r['n']:7d} {r['mae']:8.3f} "
              f"{r['rmse']:8.3f} {r['bias']:+8.3f}")
    pd.DataFrame(err_rows).to_csv("results/tables/step16_horizon_error.csv",
                                  index=False)

    # For context: the scalar predictor from step 10 only works at 10 min.
    scalar_mae = None
    try:
        scalar_mae = float(pd.read_csv("results/tables/step10_model_comparison.csv")
                           .set_index("model").loc["blend", "test_mae"])
    except Exception:
        pass
    ten = next(r for r in err_rows if r["horizon_min"] == 10)
    swing = 14.0     # typical indoor daily swing in this room, degC
    print(f"\n  at 10 min the error is {ten['mae']:.3f} degC and at 6 h it is "
          f"{err_rows[-1]['mae']:.3f} degC,")
    print(f"  which against a daily indoor swing of roughly {swing:.0f} degC is "
          f"{err_rows[-1]['mae']/swing*100:.0f}% at the six-hour horizon.")
    print(f"  The ten-minute predictor from step 10 cannot be compared here: it "
          f"was fitted on")
    print(f"  the original box, which is a different room, and it has no output "
          f"beyond ten minutes.")

    MODEL_PATH.write_text(json.dumps({
        "theta": model.theta.tolist(),
        "param_names": PARAM_NAMES,
        "fit_info": model.fit_info,
        "horizon_error": err_rows,
    }, indent=2))
    print(f"\n  wrote {MODEL_PATH}")

    # ------------------------------------------------------------------ figure
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # A held-out 48-hour trajectory, measured against predicted.
    ax = axes[0]
    df = val_sessions[0]
    ti = df["t_in"].to_numpy(float)
    to = df["t_out"].to_numpy(float)
    u = df["u"].to_numpy(float)
    f = df["f"].to_numpy(float)
    q = float(df["heater_w"].iloc[0])
    hrs = df["t_s"].to_numpy(float) / 3600.0
    ax.plot(hrs, to, color=P.INK_MUTED, lw=0.8, ls=(0, (2, 2)), zorder=2,
            label="Outdoor air")
    ax.plot(hrs, ti, color=P.SERIES[0], lw=1.0, zorder=4, label="Measured indoor air")
    # Re-predict from scratch every 6 h, so each segment is a genuine
    # six-hour-ahead forecast rather than a corrected one.
    seg = int(6 * 3600 / cfg.DT_CONTROL)
    for s in range(0, len(ti) - 1, seg):
        e = min(s + seg, len(ti) - 1)
        pi, _ = model.rollout(ti[s], df["t_wall"].to_numpy(float)[s],
                              to[s:e], u[s:e], f[s:e], q)
        ax.plot(hrs[s:e + 1], pi, color=P.SERIES[1], lw=1.0, zorder=3,
                label="Predicted, 6 h ahead" if s == 0 else None)
    ax.set_xlabel("hours into the session")
    ax.set_ylabel("°C")
    ax.set_title(f"Held-out session: {df['location'].iloc[0].replace('_',' ')}")
    ax.set_xlim(0, hrs[-1])

    ax = axes[1]
    h = [r["horizon_min"] for r in err_rows]
    ax.plot(h, [r["mae"] for r in err_rows], color=P.SERIES[0], marker="o",
            ms=4, zorder=4, label="identified two-state model")
    if scalar_mae:
        ax.plot([10], [scalar_mae], color=P.SERIES[1], marker="s", ms=5,
                ls="none", zorder=5, label="step 10 predictor (10 min only)")
    ax.set_xscale("log")
    ax.set_xticks(h, [str(v) for v in h])
    ax.set_xlabel("prediction horizon (minutes)")
    ax.set_ylabel("mean absolute error (°C)")
    ax.set_title("Error against horizon")
    ax.set_ylim(0, max(r["mae"] for r in err_rows) * 1.25)

    P.legend_below(axes[0], ncol=3, gap=32)
    P.legend_below(axes[1], ncol=1, gap=32)
    sup = fig.suptitle("Step 16  ·  Room dynamics identified from logged data",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold",
                       color=P.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    note = P.caption(axes[0],
        f"Left: one held-out 48-hour session. The prediction is restarted every "
        f"6 hours from the measured state, so each orange segment is a genuine "
        f"six-hour forecast.\\nRight: error against horizon, measured on "
        f"{len(val_sessions)} held-out sessions the fit never saw.", gap=54)
    fig.savefig("results/figures/step16_identification.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/step16_identification.png")

    checks = [
        ("the fit converged", model.fit_info["success"]),
        ("air capacity recovered within 25%", abs(c_in_hat / cfg.C_IN - 1) < 0.25),
        ("structure capacity recovered within 25%",
         abs(c_wall_hat / cfg.C_WALL - 1) < 0.25),
        ("heat loss coefficient recovered within 20%",
         abs(ua_hat / ua_true - 1) < 0.20),
        ("structural time constant recovered within 20%",
         abs(tau_hat / tau_true - 1) < 0.20),
        ("every parameter is physically valid", bool((ident > 0).all())),
        ("10-minute error below 0.5 degC", ten["mae"] < 0.5),
        ("6-hour error below 1.0 degC", err_rows[-1]["mae"] < 1.0),
        ("no systematic bias at 6 h", abs(err_rows[-1]["bias"]) < 0.5),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(c[1] for c in checks)
    print(f"\nSTEP 16 DONE-WHEN: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1

if __name__ == "__main__":
    raise SystemExit(main())
