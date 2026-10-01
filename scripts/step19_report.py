"""Step 13 (file step19): report figures, README and the web page."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.gains import solar_w, internal_w
from windwindow.realweather import LOCATIONS, build

EVAL = pathlib.Path("data/evaluate")
POLICY_ORDER = ["sealed", "open throughout", "timer", "threshold rule",
                "controller", "reference, perfect information"]
POLICY_SHORT = {"sealed": "Sealed", "open throughout": "Open throughout",
                "timer": "Timer", "threshold rule": "Threshold rule",
                "controller": "Controller",
                "reference, perfect information": "Perfect information"}
# Categorical slots are used in their canonical order, never picked out of
# sequence: the ordering is what the palette's colour-blind separation was
# validated on. The controller takes slot 1 as the subject of the report; the
# perfect-information reference is drawn in ink because it is a bound rather
# than another policy.
POLICY_COLOR = {"controller": P.SERIES[0],
                "sealed": P.SERIES[1],
                "threshold rule": P.SERIES[2],
                "open throughout": P.SERIES[3],
                "timer": P.SERIES[4],
                "reference, perfect information": P.INK_2}


def fig_climates():
    """The five real weather series, with the overnight lows that decide everything."""
    P.use_style()
    fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.6), sharey=True)
    for ax, loc in zip(axes, LOCATIONS):
        w = build(loc, hours=48, start_hour=0)
        h = w.times / 3600.0
        ax.axhspan(cfg.COMFORT_C, 50, color=P.GRID, zorder=0)
        ax.plot(h, w.t_out_grid, color=P.TEMP_COLOR["t_out"], lw=1.0, zorder=4)
        ax.axhline(cfg.COMFORT_C, color=P.SERIES[1], lw=0.8, ls=(0, (3, 2)), zorder=3)
        s = w.summary()
        # The overnight mean goes in the title. Inside the panel it collides
        # with the trace, which dips lowest in exactly the climates where the
        # number matters most.
        ax.set_title(f"{w.label.split(',')[0]}\nnight mean "
                     f"{s['night_mean']:.1f} \u00b0C", fontsize=8.5,
                     linespacing=1.5)
        ax.set_xlim(0, 48)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_xlabel("hours")
    axes[0].set_ylabel("outdoor °C")
    axes[0].set_ylim(10, 48)
    axes[0].annotate("above 26 °C", xy=(0.8, 46.5), fontsize=6.2,
                     color=P.INK_MUTED, va="top")
    sup = fig.suptitle("Real weather, 48 hours from midnight, five climates",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold",
                       color=P.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    note = P.caption(axes[0],
        "Hourly observations from the Open-Meteo archive for real heat events, "
        "interpolated to the simulation step with a small turbulent component "
        "added.\nThe overnight mean is the figure that decides whether "
        "ventilation can help at all: it is the coolest air the room will be "
        "offered.", gap=42)
    fig.savefig("results/figures/rep_climates.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/rep_climates.png")


def fig_strategy(loc="palo_alto", seed=2):
    """The daily strategy the controller finds: flush, seal, reopen."""
    P.use_style()
    need = {"controller": None, "sealed": None, "open throughout": None}
    for k in need:
        f = EVAL / f"{loc}_s{seed}_{k.replace(' ', '_')}.csv"
        if not f.exists():
            print(f"  (skipping strategy figure, {f} missing)")
            return
        need[k] = pd.read_csv(f)
    df = need["controller"]
    w = build(loc, seed=seed, hours=48, start_hour=0)
    h = df["t_s"].to_numpy(float) / 3600.0
    u = df["u"].to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.15]})
    ax = axes[0]
    lo = min(df["t_out_true"].min(), need["controller"]["t_in_true"].min()) - 1.5
    hi = max(need["sealed"]["t_in_true"].max(), df["t_out_true"].max()) + 1.5
    ax.axhspan(cfg.COMFORT_C, hi, color=P.GRID, zorder=0)
    ax.axhline(cfg.COMFORT_C, color=P.SERIES[1], lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.plot(h, df["t_out_true"], color=P.INK_MUTED, lw=0.9, ls=(0, (2, 2)),
            zorder=3, label="Outdoor air")
    ax.plot(h, need["sealed"]["t_in_true"], color=POLICY_COLOR["sealed"], lw=1.0,
            zorder=4, label="Indoor, sealed")
    ax.plot(h, need["open throughout"]["t_in_true"],
            color=POLICY_COLOR["open throughout"], lw=1.0, zorder=4,
            label="Indoor, open throughout")
    ax.plot(h, df["t_in_true"], color=POLICY_COLOR["controller"], lw=1.5,
            zorder=6, label="Indoor, controller")
    ax.set_ylabel("\u00b0C")
    ax.set_ylim(lo, hi)
    ax.set_title(f"{w.label}: what the controller does over two days")
    ax.annotate("26 \u00b0C", xy=(0.3, cfg.COMFORT_C + 0.4), fontsize=6.4,
                color=P.SERIES[1])

    # Name the phases from the trace rather than from assumed clock times.
    open_state = u > 0.05
    segs, s0 = [], 0
    for i in range(1, len(open_state) + 1):
        if i == len(open_state) or open_state[i] != open_state[s0]:
            segs.append((h[s0], h[i - 1] + (h[1] - h[0]), bool(open_state[s0])))
            s0 = i
    for a, bnd, is_open in segs:
        if bnd - a < 2.0:
            continue
        mid = (a + bnd) / 2.0
        clock = mid % 24
        if not is_open:
            name = "seal"
        elif bnd - a > 10.0:
            # One continuous opening that runs from the afternoon straight
            # through the night. Calling it either name alone would misread it.
            name = "reopen, then flush overnight"
        elif clock < 11:
            name = "flush"
        else:
            name = "reopen"
        ax.annotate(name, xy=(mid, hi - 0.8), ha="center", va="top",
                    fontsize=7, color=P.INK_2, zorder=7)
        ax.axvline(a, color=P.AXIS, lw=0.5, ls=(0, (1, 3)), zorder=1)

    ax = axes[1]
    ax.fill_between(h, 0, u, step="post", color=POLICY_COLOR["controller"],
                    alpha=0.85, lw=0, zorder=3)
    ax.plot(h, df["f"], drawstyle="steps-post", color=P.INK_2, lw=0.9,
            ls=(0, (2, 1.5)), zorder=4, label="fan speed")
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1], ["shut", "open"])
    ax.set_ylabel("window")
    ax.set_xlabel("hours from midnight")
    ax.set_xlim(0, h[-1])
    ax.set_xticks(np.arange(0, 49, 6))

    P.legend_below(axes[1], ncol=5, gap=34, source=[axes[0], axes[1]])
    note = P.caption(axes[1],
        f"Weather seed {seed}, which the controller was not tuned on. The shaded "
        f"height in the lower panel is how far the window is open.\n"
        f"Phase names are read off the trace, not assumed: the pattern repeats "
        f"daily, flushing the structure overnight while the outdoor air is "
        f"cold,\nsealing through the morning and midday so the stored coolth is "
        f"spent slowly, then reopening once the outdoor air falls back below "
        f"the room.", gap=54)
    fig.tight_layout()
    fig.savefig("results/figures/rep_strategy.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note])
    plt.close(fig)
    print("  wrote results/figures/rep_strategy.png")


def fig_by_climate(by_clim, runs):
    """Degree-minutes by climate and policy, plus what it buys per climate."""
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    piv = (runs.pivot_table(index="location", columns="policy",
                            values="degree_minutes_over_26")
               .reindex(index=list(LOCATIONS), columns=POLICY_ORDER))
    x = np.arange(len(piv))
    show = ["sealed", "threshold rule", "controller",
            "reference, perfect information"]
    w = 0.2
    for i, pol in enumerate(show):
        ax.bar(x + (i - 1.5) * w, piv[pol] / 1000.0, width=w * 0.9,
               color=POLICY_COLOR[pol], label=POLICY_SHORT[pol], zorder=4)
    ax.set_xticks(x, [LOCATIONS[k]["label"].split(",")[0] for k in piv.index],
                  fontsize=7)
    ax.set_ylabel("thousand degree-minutes above 26 °C\nover 48 hours")
    ax.set_title("Overheating by climate and policy")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    # Sorted by the value: this panel is a ranking, and a single hue carries
    # no entity identity to preserve.
    b = by_clim.set_index("location").sort_values("vs_sealed_pct")
    ax.barh(range(len(b)), b.vs_sealed_pct, height=0.6,
            color=[POLICY_COLOR["controller"] if v > 25 else P.INK_MUTED
                   for v in b.vs_sealed_pct], zorder=4)
    for i, (v, n) in enumerate(zip(b.vs_sealed_pct, b.night_mean_c)):
        ax.annotate(f"{v:.0f}%   (night {n:.0f} °C)", (max(v, 0), i),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=6.8, color=P.INK_2)
    ax.set_yticks(range(len(b)),
                  [LOCATIONS[k]["label"].split(",")[0] for k in b.index],
                  fontsize=7)
    ax.set_xlabel("reduction in overheating against a sealed room (%)")
    ax.set_title("What ventilation control buys")
    ax.set_xlim(0, max(100, b.vs_sealed_pct.max() * 1.45))
    ax.grid(axis="y", visible=False)

    P.legend_below(axes[0], ncol=4, gap=38)
    sup = fig.suptitle("Results across five real climates",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold",
                       color=P.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    note = P.caption(axes[0],
        "Mean of two weather seeds per climate, neither used to tune the "
        "controller. 'Perfect information' is the same controller given the "
        "true room\nparameters and an exact forecast, so the gap to it "
        "measures what better information would buy rather than what a "
        "different algorithm would.", gap=50)
    fig.savefig("results/figures/rep_by_climate.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/rep_by_climate.png")


def fig_ablation(ctl):
    """What each capability contributes, and the fan trade-off."""
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    abl = ctl[ctl.group == "ablation"].copy()
    full = abl[abl.label == "full controller"].iloc[0].degree_minutes_over_26
    abl = abl[abl.label != "full controller"]
    abl["cost"] = (abl.degree_minutes_over_26 / full - 1) * 100
    abl = abl.sort_values("cost")
    ax = axes[0]
    cols = [P.SERIES[1] if v > 0 else P.SERIES[2] for v in abl.cost]
    ax.barh(range(len(abl)), abl.cost, height=0.6, color=cols, zorder=4)
    for i, v in enumerate(abl.cost):
        ax.annotate(f"{v:+.1f}%", (v, i), textcoords="offset points",
                    xytext=(5 if v >= 0 else -5, 0), va="center",
                    ha="left" if v >= 0 else "right", fontsize=7, color=P.INK_2)
    ax.axvline(0, color=P.AXIS, lw=0.8, zorder=5)
    ax.set_yticks(range(len(abl)), list(abl.label), fontsize=7)
    ax.set_xlabel("change in overheating against the full controller (%)")
    ax.set_title("Effect of changing one thing")
    ax.set_xlim(min(0, abl.cost.min() * 1.3), abl.cost.max() * 1.45)
    ax.grid(axis="y", visible=False)

    sw = ctl[ctl.group == "sweep"].sort_values("fan_wh")
    ax = axes[1]
    ax.plot(sw.fan_wh / 1000.0, sw.degree_minutes_over_26 / 1000.0,
            color=P.SERIES[0], marker="o", ms=4, zorder=4,
            label="controller, fan penalty varied")
    rule = ctl[ctl.label == "threshold rule"].iloc[0]
    ax.plot([rule.fan_wh / 1000.0], [rule.degree_minutes_over_26 / 1000.0],
            marker="s", ms=6, ls="none", color=POLICY_COLOR["threshold rule"],
            zorder=5, label="threshold rule")
    ax.set_xlabel("fan electricity over 48 h (kWh)")
    ax.set_ylabel("thousand degree-minutes above 26 °C")
    ax.set_title("Comfort against fan energy")

    P.legend_below(axes[1], ncol=2, gap=34)
    sup = fig.suptitle("Where the result comes from",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold",
                       color=P.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    note = P.caption(axes[0],
        "Left: each row changes one thing about the full controller and re-runs "
        "it, so a positive number means the change made the room warmer. Four "
        "of the five remove a\ncapability; 'perfect forecast' instead hands it "
        "an exact one, and changes nothing, which is the point.\n"
        "Right: the threshold rule reaches similar comfort but sits far to the "
        "right, because it runs the fan whenever the window is open rather "
        "than only when the fan earns its electricity.", gap=56)
    fig.savefig("results/figures/rep_ablation.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/rep_ablation.png")


def fig_gains():
    """The daily heat-gain profile, which sets the timing of everything."""
    P.use_style()
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    h = np.arange(0, 24.01, 0.05)
    ax.fill_between(h, 0, internal_w(h), color=P.SERIES[3], alpha=0.9, lw=0,
                    zorder=3, label="occupants and equipment")
    ax.fill_between(h, internal_w(h), internal_w(h) + solar_w(h),
                    color=P.SERIES[1], alpha=0.9, lw=0, zorder=3,
                    label="sun through the window")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("heat gain (W)")
    ax.set_title("Daily heat gains in the simulated room")
    P.legend_below(ax, ncol=2, gap=34)
    note = P.caption(ax,
        "Solar gain peaks in the late afternoon on a west-facing window and "
        "occupancy peaks in the evening, so the room is loaded\nhardest when "
        "the outdoor air is least able to take the heat away. Both are "
        "predictable from the clock, so the controller is\nallowed to know "
        "them; the weather is not.", gap=52)
    fig.tight_layout()
    fig.savefig("results/figures/rep_gains.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note])
    plt.close(fig)
    print("  wrote results/figures/rep_gains.png")


def main():
    for d in ("results/figures", "results/tables"):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv("results/tables/step18_runs.csv")
    by_clim = pd.read_csv("results/tables/step18_by_climate.csv")
    ctl = pd.read_csv("results/tables/step17_controller.csv")

    fig_climates()
    fig_gains()
    fig_strategy()
    fig_by_climate(by_clim, runs)
    fig_ablation(ctl)

    figs = sorted(pathlib.Path("results/figures").glob("*.png"))
    tables = sorted(pathlib.Path("results/tables").glob("*.csv"))
    print(f"\n  {len(figs)} figures, {len(tables)} tables")

    from windwindow.report2 import write_readme, write_html
    write_readme(runs, by_clim, ctl, figs, tables)
    write_html(runs, by_clim, ctl, figs, tables)

    checks = [
        ("all report figures written",
         all(pathlib.Path(f"results/figures/rep_{n}.png").exists()
             for n in ("climates", "gains", "strategy", "by_climate", "ablation"))),
        ("README written", pathlib.Path("README.md").exists()),
        ("index.html written and fully filled in",
         pathlib.Path("index.html").exists()
         and "{{" not in pathlib.Path("index.html").read_text()),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(c[1] for c in checks)
    print(f"\nSTEP 13 DONE-WHEN: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
