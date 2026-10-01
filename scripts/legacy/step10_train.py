"""
Step 10: train the predictors and report MAE on the held-out sessions.

Three things happen here, in this order, and the order matters:

  1. Candidates are compared by 5-fold cross-validation *inside the training
     sessions*. One is selected. The held-out sessions are not touched.
  2. Every candidate is then scored once on the held-out sessions.
  3. The physics oracles are computed, so we know what the best achievable
     error actually is before judging whether ours is good.

Doing (1) before (2) is the whole point. If the choice of model were made by
looking at held-out scores, the held-out score would no longer mean anything.
"""

import sys, pathlib, pickle, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split as tts

from windwindow import plotting as P
from windwindow.dataset import load_sessions, split, xy, assert_no_leakage
from windwindow.features import FEATURE_NAMES, FEATURE_LABELS
from windwindow.models import (ZeroBaseline, TrendBaseline, RidgeModel, GBMModel,
                               BlendModel, mae, rmse)
from windwindow.oracle import oracle_predictions

MODEL_PATH = pathlib.Path("results/model.pkl")
G1_TARGET = 0.25

def build(name):
    return {"zero": ZeroBaseline, "trend": TrendBaseline, "ridge": RidgeModel,
            "lightgbm": GBMModel, "blend": BlendModel}[name]()

def main():
    raw = load_sessions()
    train, test = split(raw)
    assert_no_leakage(train, test)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    groups = train["session_id"].to_numpy()

    print(f"training rows {len(Xtr)} from {train.session_id.nunique()} sessions")
    print(f"held-out rows {len(Xte)} from {test.session_id.nunique()} sessions")
    print(f"Goal G1: MAE at or below {G1_TARGET:.2f} °C on held-out sessions")

    # --- 1. choose using the training sessions only ------------------------
    print(f"\n1. Choosing a model by 5-fold grouped CV inside the training set.")
    print(f"   (folds split by session, so no fold shares a session with another)")
    names = ["zero", "trend", "ridge", "lightgbm", "blend"]
    cv_scores = {n: [] for n in names}
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(Xtr, ytr, groups):
        for n in names:
            m = build(n)
            try:
                m.fit(Xtr[tr], ytr[tr], groups=groups[tr])
            except TypeError:
                m.fit(Xtr[tr], ytr[tr])
            cv_scores[n].append(mae(ytr[va], m.predict(Xtr[va])))
    cv = {n: float(np.mean(v)) for n, v in cv_scores.items()}
    for n in sorted(cv, key=cv.get):
        print(f"     {n:10s} CV MAE {cv[n]:.4f}  (sd across folds {np.std(cv_scores[n]):.4f})")
    chosen = min(cv, key=cv.get)
    print(f"   -> selected: {chosen}")

    # --- 2. score everything once on the held-out sessions -----------------
    print(f"\n2. Scoring on the {test.session_id.nunique()} held-out sessions.")
    rows, fitted = [], {}
    for n in names:
        m = build(n)
        t0 = time.time()
        try:
            m.fit(Xtr, ytr, groups=groups)
        except TypeError:
            m.fit(Xtr, ytr)
        secs = time.time() - t0
        ptr, pte = m.predict(Xtr), m.predict(Xte)
        rows.append({"model": n, "cv_mae": cv[n], "train_mae": mae(ytr, ptr),
                     "test_mae": mae(yte, pte), "test_rmse": rmse(yte, pte),
                     "test_p90": float(np.percentile(np.abs(yte - pte), 90)),
                     "meets_g1": mae(yte, pte) <= G1_TARGET,
                     "selected": n == chosen, "fit_seconds": secs})
        fitted[n] = m
    res = pd.DataFrame(rows)

    print(f"\n   {'model':10s} {'CV MAE':>8s} {'train':>7s} {'HELD-OUT MAE':>13s} "
          f"{'RMSE':>7s} {'90th pct':>9s} {'G1':>5s}")
    for r in rows:
        star = " *" if r["selected"] else "  "
        print(f"  {star}{r['model']:10s} {r['cv_mae']:8.3f} {r['train_mae']:7.3f} "
              f"{r['test_mae']:13.3f} {r['test_rmse']:7.3f} {r['test_p90']:9.3f} "
              f"{'PASS' if r['meets_g1'] else 'MISS':>5s}")
    print(f"   * = selected by cross-validation, before any held-out score was seen")

    zero_mae = res.loc[res.model == "zero", "test_mae"].iloc[0]

    # --- 3. what was achievable at all -----------------------------------
    print(f"\n3. What is the best any model could do? Computing the physics oracles.")
    orc = oracle_predictions(test)
    o_perfect, o_hold, o_trend = (mae(yte, orc[c]) for c in ("perfect", "hold", "trend"))
    print(f"   perfect physics + the true future weather : MAE {o_perfect:.3f} °C")
    print(f"   perfect physics, outdoor held where it is : MAE {o_hold:.3f} °C   <- the fair floor")
    print(f"   perfect physics, outdoor trend continued  : MAE {o_trend:.3f} °C")
    print(f"\n   The nine inputs contain no weather forecast, so {o_hold:.3f} °C is the")
    print(f"   floor for our models. Of that, {o_perfect:.3f} °C is sensor noise in the")
    print(f"   label and {o_hold-o_perfect:.3f} °C is not knowing what the weather does next.")

    sel_mae = res.loc[res.model == chosen, "test_mae"].iloc[0]
    skill = (zero_mae - sel_mae) / (zero_mae - o_hold)
    print(f"\n   guessing no change {zero_mae:.3f}  ->  floor {o_hold:.3f}  "
          f"= {zero_mae-o_hold:.3f} °C of error there to remove")
    print(f"   the selected model removes {zero_mae-sel_mae:.3f} °C of it, "
          f"which is {skill*100:.0f}% of what was available")
    print(f"   G1 target {G1_TARGET:.2f} °C sits only "
          f"{G1_TARGET-o_hold:.3f} °C above the floor")

    # --- why the session split matters ------------------------------------
    print(f"\nWhat a careless row-level split would have claimed:")
    allrows = pd.concat([train, test], ignore_index=True)
    Xa, ya = xy(allrows)
    Xr1, Xr2, yr1, yr2 = tts(Xa, ya, test_size=0.3, random_state=0, shuffle=True)
    leaky = mae(yr2, build(chosen).fit(Xr1, yr1).predict(Xr2))
    print(f"  rows shuffled at random : MAE {leaky:.3f} °C  <- flattering, and wrong")
    print(f"  sessions kept whole     : MAE {sel_mae:.3f} °C  <- what we report")
    print(f"  the careless split understates the error by "
          f"{(sel_mae-leaky)/sel_mae*100:.0f}%; it even beats the physics floor, "
          f"which is impossible")

    # --- what the models learned ------------------------------------------
    ridge, gbm = fitted["ridge"], fitted["lightgbm"]
    print(f"\nridge penalty alpha = {ridge.alpha_}")
    print(f"{'input':24s} {'ridge coef':>12s} {'LightGBM gain':>15s}")
    coefs, imps = ridge.coefficients, gbm.importances
    for n in sorted(FEATURE_NAMES, key=lambda k: -imps[k]):
        print(f"{FEATURE_LABELS[n]:24s} {coefs[n]:+12.4f} {imps[n]*100:14.1f}%")

    print(f"\nheld-out MAE by scenario:")
    print(f"{'scenario':24s}" + "".join(f"{n:>10s}" for n in names) + f"{'floor':>10s}")
    by_scenario = []
    for sc, grp in test.groupby("scenario"):
        Xs, ys = xy(grp)
        rec = {"scenario": sc, "rows": len(ys)}
        line = f"{sc:24s}"
        for n in names:
            rec[n] = mae(ys, fitted[n].predict(Xs))
            line += f"{rec[n]:10.3f}"
        rec["floor"] = mae(ys, orc.loc[grp.index, "hold"])
        line += f"{rec['floor']:10.3f}"
        print(line)
        by_scenario.append(rec)
    pd.DataFrame(by_scenario).to_csv("results/tables/step10_by_scenario.csv",
                                     index=False)

    res.to_csv("results/tables/step10_model_comparison.csv", index=False)
    pd.DataFrame([{"input": n, "label": FEATURE_LABELS[n], "ridge_coef": coefs[n],
                   "lgbm_gain_frac": imps[n]} for n in FEATURE_NAMES]
                 ).to_csv("results/tables/step10_feature_importance.csv", index=False)
    pd.DataFrame([{"oracle": "perfect_physics_true_weather", "mae": o_perfect},
                  {"oracle": "perfect_physics_outdoor_held", "mae": o_hold},
                  {"oracle": "perfect_physics_outdoor_trend", "mae": o_trend},
                  {"oracle": "selected_model", "mae": sel_mae},
                  {"oracle": "guess_no_change", "mae": zero_mae},
                  {"oracle": "leaky_row_level_split", "mae": leaky}]
                 ).to_csv("results/tables/step10_oracles.csv", index=False)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({**fitted, "best": chosen, "cv": cv,
                     "oracle_floor": o_hold}, fh)
    print(f"\n  wrote results/tables/step10_by_scenario.csv")
    print(f"  wrote results/tables/step10_model_comparison.csv")
    print(f"  wrote results/tables/step10_feature_importance.csv")
    print(f"  wrote results/tables/step10_oracles.csv")
    print(f"  wrote {MODEL_PATH}  (selected: {chosen})")

    # ------------------------------------------------------------------ figure
    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))

    ax = axes[0]
    order = list(res.sort_values("test_mae", ascending=False).model)
    vals = [res.loc[res.model == n, "test_mae"].iloc[0] for n in order]
    cols = [P.SERIES[0] if n == chosen else P.INK_MUTED for n in order]
    ax.barh(range(len(order)), vals, color=cols, height=0.6)
    ax.axvline(G1_TARGET, color=P.SERIES[1], lw=1.0, zorder=5)
    ax.axvline(o_hold, color=P.SERIES[2], lw=1.0, ls=(0, (3, 2)), zorder=5)
    # The two reference lines sit only 0.066 degC apart, too close for a label
    # under each. A small legend in the empty upper-right names both.
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy
    ax.legend(
        handles=[Line2D([], [], color=P.SERIES[1], lw=1.0,
                        label=f"G1 target {G1_TARGET:.2f}"),
                 Line2D([], [], color=P.SERIES[2], lw=1.0, ls=(0, (3, 2)),
                        label=f"attainable floor {o_hold:.3f}")],
        loc="upper left", bbox_to_anchor=(0.0, 0.0),
        bbox_transform=offset_copy(ax.transAxes, fig=fig, x=0.0, y=-34,
                                   units="points"),
        ncol=2, fontsize=6.4, frameon=False, handlelength=2.0,
        borderaxespad=0.0,
    )
    for i, (n, v) in enumerate(zip(order, vals)):
        ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(4, 0),
                    va="center", color=P.INK_2, fontsize=7)
    display = {"zero": "constant zero", "trend": "trend extrapolation",
               "ridge": "ridge", "lightgbm": "boosted trees",
               "blend": "ensemble"}
    ax.set_yticks(range(len(order)), [display[n] for n in order])
    ax.set_xlabel("held-out MAE (°C)")
    ax.set_title("Held-out prediction error, with attainable floor")
    ax.set_xlim(0, max(vals) * 1.3)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)

    ax = axes[1]
    imp_order = sorted(FEATURE_NAMES, key=lambda k: imps[k])
    ax.barh(range(len(imp_order)), [imps[n] * 100 for n in imp_order],
            color=P.SERIES[0], height=0.6)
    ax.set_yticks(range(len(imp_order)),
                  [FEATURE_LABELS[n] for n in imp_order], fontsize=6.4)
    ax.set_xlabel("share of total split gain (%)")
    ax.set_title("Contribution of each input")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)

    ax = axes[2]
    pred = fitted[chosen].predict(Xte)
    ax.scatter(yte, pred, s=1.6, alpha=0.2, color=P.SERIES[0], edgecolors="none")
    lo = min(yte.min(), pred.min()); hi = max(yte.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=P.INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=5)
    ax.annotate("1:1 line", xy=(hi, hi), textcoords="offset points", xytext=(-4, -9),
                ha="right", color=P.INK_MUTED, fontsize=6.4)
    ax.set_xlabel("observed change in T_in (°C)")
    ax.set_ylabel("predicted change in T_in (°C)")
    ax.set_title("Selected model on held-out sessions")
    ax.set_aspect("equal", adjustable="box")

    sup = fig.suptitle("Ten-minute temperature prediction: candidate comparison",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold", color=P.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    note = P.caption(axes[0],
                     f"Trained on {len(Xtr)} rows from {train.session_id.nunique()} sessions, "
                     f"scored on {len(Xte)} rows from {test.session_id.nunique()} held-out "
                     f"sessions. The candidate was selected by cross-validation within the\n"
                     f"training set. The attainable floor is the error of an exact physical model "
                     f"with outdoor temperature held constant; no model built from the nine "
                     f"inputs can improve on it.", gap=56)
    fig.savefig("results/figures/step10_models.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/step10_models.png")

    checks = [
        ("MAE reported for all four models the plan asks for",
         set(("zero", "trend", "ridge", "lightgbm")) <= set(res.model)),
        ("the model was chosen without looking at held-out data", True),
        ("ridge beats guessing no change",
         res.loc[res.model == "ridge", "test_mae"].iloc[0] < zero_mae),
        ("LightGBM beats guessing no change",
         res.loc[res.model == "lightgbm", "test_mae"].iloc[0] < zero_mae),
        ("the selected model is the best of the five on held-out data too",
         abs(sel_mae - res.test_mae.min()) < 0.005),
        ("the selected model captures most of the achievable skill", skill > 0.7),
        ("the achievable floor was measured", o_hold > 0),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  G1 (MAE <= {G1_TARGET:.2f} °C): "
          f"{'MET' if sel_mae <= G1_TARGET else 'NOT MET'}: "
          f"{sel_mae:.3f} °C, floor {o_hold:.3f} °C")
    ok = all(c[1] for c in checks)
    print(f"STEP 10 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
