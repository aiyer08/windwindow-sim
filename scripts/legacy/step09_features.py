"""Step 9: build features and the split. Done when no test session leaks into training."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windwindow import plotting as P
from windwindow.dataset import load_sessions, usable, split, xy, assert_no_leakage
from windwindow.features import FEATURE_NAMES, FEATURE_LABELS

def main():
    raw = load_sessions()
    use = usable(raw)
    train, test = split(raw)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)

    print(f"rows logged                     {len(raw)}")
    print(f"rows with a stable action        {len(use)}  ({len(use)/len(raw)*100:.1f}%)")
    print(f"  -- the rest are dropped because u or f changed inside the 10 min window,")
    print(f"     or the row is in the first 5 min and has no trend history yet")
    print(f"\ntraining rows  {len(train):6d}  from {train.session_id.nunique()} sessions")
    print(f"held-out rows  {len(test):6d}  from {test.session_id.nunique()} sessions")

    overlap = assert_no_leakage(train, test)
    print(f"\nsessions in both splits: {len(overlap)}  <- must be 0")

    # Show *why* the session split matters: neighbouring rows are near-copies.
    one = raw[raw.session_id == raw.session_id.iloc[0]]
    r1 = float(one["t_in"].autocorr(lag=1))
    r20 = float(one["t_in"].autocorr(lag=20))
    print(f"\nwithin one session, T_in autocorrelation at lag 1 (30 s):  {r1:.4f}")
    print(f"                                        at lag 20 (10 min): {r20:.4f}")
    print(f"  -> a random row-level split would put near-duplicates of almost every")
    print(f"     test row into the training set. Step 10 measures what that costs.")

    print(f"\nlabel (change in T_in over 10 min), training rows:")
    print(f"  mean {ytr.mean():+.3f}   sd {ytr.std():.3f}   "
          f"range {ytr.min():+.2f} .. {ytr.max():+.2f} degC")
    print(f"  a model that always guesses zero would score "
          f"MAE {np.abs(ytr).mean():.3f} degC on training rows")

    print(f"\n{'feature':22s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'max':>8s} "
          f"{'corr with label':>16s}")
    stats = []
    for i, n in enumerate(FEATURE_NAMES):
        c = np.corrcoef(Xtr[:, i], ytr)[0, 1] if Xtr[:, i].std() > 0 else np.nan
        print(f"{FEATURE_LABELS[n]:22s} {Xtr[:,i].mean():8.3f} {Xtr[:,i].std():8.3f} "
              f"{Xtr[:,i].min():8.2f} {Xtr[:,i].max():8.2f} {c:16.3f}")
        stats.append({"feature": n, "label": FEATURE_LABELS[n], "mean": Xtr[:, i].mean(),
                      "sd": Xtr[:, i].std(), "min": Xtr[:, i].min(), "max": Xtr[:, i].max(),
                      "corr_with_label": c})
    pd.DataFrame(stats).to_csv("results/tables/step09_feature_stats.csv", index=False)

    # Per-scenario and per-action balance of the training set.
    print(f"\ntraining rows by scenario:")
    print(train.groupby("scenario").size().to_string())
    print(f"\ntraining rows by window position:")
    print(train.groupby(train.u.round(2)).size().to_string())

    # ------------------------------------------------------------------ figure
    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8))

    ax = axes[0]
    bins = np.linspace(min(ytr.min(), yte.min()), max(ytr.max(), yte.max()), 45)
    ax.hist(ytr, bins=bins, color=P.SERIES[0], label=f"training ({len(ytr)})")
    ax.hist(yte, bins=bins, histtype="step", lw=1.0, color=P.SERIES[1],
            label=f"held out ({len(yte)})")
    ax.set_title("What we are predicting")
    ax.set_xlabel("change in T_in over 10 min (°C)")
    ax.set_ylabel("rows")

    ax = axes[1]
    # The feature that carries the most signal, against the label.
    idx = int(np.nanargmax(np.abs([np.corrcoef(Xtr[:, i], ytr)[0, 1]
                                   for i in range(len(FEATURE_NAMES))])))
    ax.scatter(Xtr[:, idx], ytr, s=1.6, color=P.SERIES[0], alpha=0.25,
               edgecolors="none", label="training row")
    ax.set_title("Strongest single input")
    ax.set_xlabel(FEATURE_LABELS[FEATURE_NAMES[idx]] + "   (strongest of the nine)")
    ax.set_ylabel("change in T_in (°C)")
    ax.axhline(0, color=P.AXIS, lw=0.5, zorder=1)

    ax = axes[2]
    # Coverage: where in (outdoor gap, window position) space do we have data?
    counts = (train.assign(dbin=pd.cut(train.d, np.arange(-10, 11, 2)),
                           ubin=train.u.round(2))
                   .groupby(["ubin", "dbin"], observed=True).size().unstack(fill_value=0))
    im = ax.imshow(counts.to_numpy(), aspect="auto", cmap="Blues", origin="lower")
    ax.set_xticks(range(len(counts.columns)),
                  [f"{iv.left:g}" for iv in counts.columns], fontsize=6)
    ax.set_yticks(range(len(counts.index)), [f"{v:g}" for v in counts.index], fontsize=6)
    ax.set_xlabel("T_out - T_in (°C)")
    ax.set_ylabel("window position u")
    ax.set_title("Training coverage")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6, colors=P.INK_MUTED)
    cb.outline.set_visible(False)
    cb.set_label("rows", color=P.INK_2, fontsize=7)

    P.legend_below(axes[0], ncol=2, gap=34)
    P.legend_below(axes[1], ncol=1, gap=34)
    sup = fig.suptitle("Step 9  ·  Model inputs, labels and the session-level split",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold", color=P.INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    note = P.caption(axes[0],
                     f"{len(train)} training rows from {train.session_id.nunique()} sessions; "
                     f"{len(test)} held-out rows from {test.session_id.nunique()} sessions. "
                     f"No session appears in both.", gap=54)
    fig.savefig("results/figures/step09_features.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("\n  wrote results/figures/step09_features.png")

    checks = [
        ("no session in both splits", len(overlap) == 0),
        ("enough training rows", len(train) > 1500),
        ("enough held-out rows", len(test) > 400),
        ("all 9 features present", all(n in train.columns for n in FEATURE_NAMES)),
        ("no NaNs in features or label",
         not (np.isnan(Xtr).any() or np.isnan(ytr).any()
              or np.isnan(Xte).any() or np.isnan(yte).any())),
        ("label spans both warming and cooling", ytr.min() < -1 and ytr.max() > 1),
        ("all four scenarios in training", train.scenario.nunique() == 4),
        ("all four scenarios held out", test.scenario.nunique() == 4),
        ("shut and fully-open rows both present",
         bool((train.u < 0.05).any() and (train.u > 0.95).any())),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 9 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
