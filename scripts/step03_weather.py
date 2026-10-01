"""Step 3: do all four weather scenarios generate and plot correctly?"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.weather import SCENARIOS, make_scenario
from windwindow.room import steady_state

def main():
    P.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6), sharex=True)
    ok = True
    print(f"{'scenario':24s} {'T_out range':>16s} {'slope_out max':>14s}  start (T_in/T_wall)")
    for ax, name in zip(axes.ravel(), SCENARIOS):
        # Three seeds per panel, to show the spread the gusts produce.
        for k, seed in enumerate((1, 2, 3)):
            sc = make_scenario(name, seed=seed)
            t = sc.times / 60.0
            ax.plot(t, sc.t_out_grid, color=P.TEMP_COLOR["t_out"],
                    alpha=1.0 if k == 0 else 0.35, zorder=4 - k)
        sc = make_scenario(name, seed=1)
        # Where a closed room would end up, for context.
        ss = steady_state(0, 0, float(np.mean(sc.t_out_grid)), sc.heater_w)[0]
        ax.axhline(cfg.COMFORT_C, color=P.AXIS, lw=0.5, ls=(0, (3, 3)), zorder=1)

        g = sc.t_out_grid
        n = int(cfg.SLOPE_WINDOW_S / cfg.DT_PHYSICS)
        slope = (g[n:] - g[:-n]) / (cfg.SLOPE_WINDOW_S / 60.0)   # K per min
        print(f"{name:24s} {g.min():6.2f}..{g.max():5.2f} {np.abs(slope).max():13.3f}  "
              f"{sc.t_in0:5.2f} / {sc.t_wall0:5.2f}   (closed room -> ~{ss:.1f} degC)")

        ax.set_title(P.SCENARIO_LABEL[name])
        ax.set_ylim(15.5, 36.5)
        ax.set_xlim(0, t[-1])
        ax.annotate(sc.note, xy=(0.02, 0.04), xycoords="axes fraction",
                    color=P.INK_MUTED, fontsize=6.2, va="bottom")

        # Every scenario must be finite, in a plausible outdoor band, and vary.
        ok &= bool(np.isfinite(g).all() and 10 < g.min() and g.max() < 40 and g.std() > 0.2)

    for ax in axes[1]:
        ax.set_xlabel("minutes")
    for ax in axes[:, 0]:
        ax.set_ylabel("outdoor °C")

    axes[0, 0].annotate("26 °C threshold", xy=(4, cfg.COMFORT_C),
                        textcoords="offset points", xytext=(0, 4),
                        color=P.INK_MUTED, fontsize=6.2)
    fig.suptitle("Outdoor temperature profiles for the four scenarios",
                 x=0.005, ha="left", fontsize=9.5, fontweight="bold", color=P.INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(0.005, 0.0, "Solid line: seed 1. Faded lines: seeds 2 and 3, showing the "
                         "spread contributed by the fluctuation component. Dashed: the 26 °C threshold.",
             color=P.INK_MUTED, fontsize=6.6, va="bottom")
    fig.savefig("results/figures/step03_scenarios.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("  wrote results/figures/step03_scenarios.png")

    print("\nSTEP 3 DONE-WHEN:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
