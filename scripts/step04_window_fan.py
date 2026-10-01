"""
Step 4: window and fan effects.

Done when opening the window cools the room in cool weather and warms it in
hot weather. Also measures the fan's marginal value and the hot-wall rebound,
because those two are what the controller in Step 11 has to reason about.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.room import Room, steady_state, rollout_exact

ACTIONS = {"shut": (0.0, 0.0), "open": (1.0, 0.0), "open_fan": (1.0, 1.0)}

def run(t_in0, t_wall0, t_out, u, f, minutes, heater=cfg.HEATER_W):
    room = Room(t_in=t_in0, t_wall=t_wall0, heater_w=heater)
    ti = [room.t_in]
    for _ in range(int(minutes * 60 / cfg.DT_PHYSICS)):
        room.step(u, f, t_out)
        ti.append(room.t_in)
    return np.arange(len(ti)) * cfg.DT_PHYSICS / 60.0, np.array(ti), room

def main():
    P.use_style()
    checks = []

    # ---------------------------------------------------------------- panel A
    # Cool weather: the room is sitting at its closed steady state, well above
    # the outdoor temperature. Opening must cool it.
    t_out_cool = 18.0
    ss = steady_state(0, 0, t_out_cool, cfg.HEATER_W)
    A = {k: run(ss[0], ss[1], t_out_cool, *ACTIONS[k], 30) for k in ACTIONS}
    print(f"A. Cool weather, T_out={t_out_cool}, room starts at its closed steady "
          f"state {ss[0]:.2f} degC")
    for k in ACTIONS:
        print(f"     {P.ACTION_LABEL[k]:14s} after 30 min: {A[k][1][-1]:6.2f} degC "
              f"({A[k][1][-1]-ss[0]:+.2f} K)")
    cools = A["open"][1][-1] < ss[0] - 0.5 and A["open_fan"][1][-1] < ss[0] - 0.5
    checks.append(("opening COOLS in cool weather", cools))

    # ---------------------------------------------------------------- panel B
    # Hot weather: the room is still cool from overnight and it is hotter
    # outside. Opening must warm it.
    t_out_hot, t_in0, t_wall0 = 31.0, 24.5, 24.0
    B = {k: run(t_in0, t_wall0, t_out_hot, *ACTIONS[k], 45) for k in ACTIONS}
    h10 = int(cfg.HORIZON_S / cfg.DT_PHYSICS)      # index at the 10-minute horizon
    print(f"\nB. Hot weather, T_out={t_out_hot}, room starts cool at {t_in0} degC")
    print(f"     {'':14s} {'at 10 min':>12s} {'at 45 min':>12s}")
    for k in ACTIONS:
        print(f"     {P.ACTION_LABEL[k]:14s} {B[k][1][h10]:9.2f} degC {B[k][1][-1]:9.2f} degC")
    # Over the 10-minute horizon the project actually decides on, opening must
    # warm the room and must warm it faster than leaving it shut.
    warms = (B["open"][1][h10] > t_in0 + 0.5 and
             B["open"][1][h10] > B["shut"][1][h10] and
             B["open_fan"][1][h10] > B["shut"][1][h10])
    # Past some point the shut room overtakes, because the heater has nowhere
    # to dump its 17.6 W. Worth knowing where that crossover sits.
    diff = B["open_fan"][1] - B["shut"][1]
    cross = np.argmax(diff < 0) * cfg.DT_PHYSICS / 60.0 if (diff < 0).any() else None
    if cross:
        print(f"     opening warms it faster for the first {cross:.0f} min; after that the")
        print(f"     shut room overtakes, because the heater has nowhere to dump 17.6 W")
    checks.append(("opening WARMS in hot weather (10 min horizon)", warms))

    # ---------------------------------------------------------------- panel C
    # The hot-wall rebound. Flush a baking room for 8 minutes, then either
    # shut the window or keep it open.
    t_out_r, ti_r, tw_r = 24.5, 28.5, 34.0
    room = Room(t_in=ti_r, t_wall=tw_r, heater_w=cfg.HEATER_W)
    ti_hist = [room.t_in]
    for _ in range(int(8 * 60)):
        room.step(1.0, 1.0, t_out_r)
        ti_hist.append(room.t_in)
    flushed_t, flushed_w = room.t_in, room.t_wall
    tail = {}
    for k in ("shut", "open_fan"):
        r2 = room.copy()
        h = []
        for _ in range(int(40 * 60)):
            r2.step(*ACTIONS[k], t_out_r)
            h.append(r2.t_in)
        tail[k] = np.array(h)
    print(f"\nC. Hot-wall rebound. Start T_in={ti_r}, T_wall={tw_r}, T_out={t_out_r}")
    print(f"     after an 8 min flush: T_in={flushed_t:.2f}, T_wall={flushed_w:.2f} "
          f"(walls still {flushed_w-flushed_t:+.2f} K above the air)")
    print(f"     then SHUT the window:  T_in climbs back to {tail['shut'].max():.2f} degC")
    print(f"     then KEEP it open:     T_in stays at     {tail['open_fan'][-1]:.2f} degC")
    rebound = tail["shut"].max() - flushed_t
    print(f"     rebound penalty for closing too early: {rebound:+.2f} K")
    checks.append(("hot walls re-warm the air after closing", rebound > 1.0))

    # ------------------------------------------------------- fan value table
    print("\nD. Is the fan worth it? 10-minute cooling, window open, fan off vs on")
    print(f"     {'T_wall-T_in':>12s} {'T_out-T_in':>11s} {'open only':>10s} {'open+fan':>9s} {'fan gains':>10s}")
    fan_rows = []
    for dw in (-3.0, 0.0, +3.0, +6.0):
        for dout in (-6.0, -3.0, -1.0):
            ti = 29.0
            a = rollout_exact(ti, ti + dw, 1, 0, ti + dout, cfg.HEATER_W, cfg.HORIZON_S)[0]
            b = rollout_exact(ti, ti + dw, 1, 1, ti + dout, cfg.HEATER_W, cfg.HORIZON_S)[0]
            print(f"     {dw:+12.1f} {dout:+11.1f} {a-ti:+10.2f} {b-ti:+9.2f} {a-b:+10.2f}")
            fan_rows.append(a - b)
    print(f"     -> the fan's value swings from {min(fan_rows):+.2f} to {max(fan_rows):+.2f} K, "
          f"a {max(fan_rows)/max(min(fan_rows),1e-9):.1f}x spread")
    checks.append(("the fan's marginal value varies a lot", max(fan_rows) > 2 * min(fan_rows)))

    # ------------------------------------------------------------- the figure
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))

    for ax, data, t_out, title, sub in (
        (axes[0], A, t_out_cool, "Cool outside: ventilation cools",
         f"T_out {t_out_cool:.0f} °C, room at its shut steady state"),
        (axes[1], B, t_out_hot, "Hot outside: ventilation warms",
         f"T_out {t_out_hot:.0f} °C, room starts cool at {t_in0:.1f} °C"),
    ):
        ax.axhline(t_out, color=P.AXIS, lw=0.5, ls=(0, (3, 3)), zorder=1)
        ax.annotate("outdoors", xy=(0.4, t_out), textcoords="offset points",
                    xytext=(0, 3), color=P.INK_MUTED, fontsize=6.2)
        # Nudge labels apart where the curves converge.
        nudge = {"shut": 4, "open": 0, "open_fan": -6} if data is B else {k: 0 for k in ACTIONS}
        for k in ("shut", "open", "open_fan"):
            t, y, _ = data[k]
            ax.plot(t, y, color=P.ACTION_COLOR[k], label=P.ACTION_LABEL[k], zorder=4)
            P.direct_label(ax, t[-1], y[-1], P.ACTION_LABEL[k], P.ACTION_COLOR[k],
                           dx=5, dy=nudge[k])
        ax.set_title(title)
        ax.set_xlabel("minutes")
        ax.set_xlim(0, 45 if data is B else 30)
        ax.axvline(cfg.HORIZON_S / 60.0, color=P.AXIS, lw=0.5, ls=(0, (1, 2)), zorder=1)

    ax = axes[2]
    t_all = np.arange(len(ti_hist)) * cfg.DT_PHYSICS / 60.0
    ax.axvspan(0, 8, color=P.GRID, zorder=0)
    ax.annotate("8 min\nflush", xy=(0.085, 0.72), xycoords="axes fraction",
                color=P.INK_MUTED, fontsize=6.2, ha="center", va="center")
    ax.plot(t_all, ti_hist, color=P.INK_2, lw=1.0, zorder=3)
    for k in ("shut", "open_fan"):
        tt = 8.0 + np.arange(1, len(tail[k]) + 1) * cfg.DT_PHYSICS / 60.0
        short = {"shut": "shut again", "open_fan": "left open"}[k]
        ax.plot(tt, tail[k], color=P.ACTION_COLOR[k], label=short, zorder=4)
        P.direct_label(ax, tt[-1], tail[k][-1], short, P.ACTION_COLOR[k], dx=5)
    ax.axhline(t_out_r, color=P.AXIS, lw=0.5, ls=(0, (3, 3)), zorder=1)
    ax.set_title("Warm shell: early closure causes rebound")
    ax.set_xlabel("minutes")
    ax.set_xlim(0, 48)
    axes[0].set_ylabel("indoor air °C")

    for ax in axes:
        ax.margins(x=0.30, y=0.16)
        ax.set_xlim(left=0)
    # Leave room on the right of every panel for the end-of-line labels.
    for ax, span in zip(axes, (30, 45, 48)):
        ax.set_xlim(0, span * 1.42)
    P.legend_below(axes[0], ncol=3, gap=42)
    P.legend_below(axes[2], ncol=2, gap=42)
    sup = fig.suptitle("Measured effect of the window and fan",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold", color=P.INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    note = P.caption(
        axes[0],
        f"Left: T_out {t_out_cool:.0f} °C, room starting at its shut steady state.    "
        f"Middle: T_out {t_out_hot:.0f} °C, room starting cool at {t_in0:.1f} °C; "
        f"dotted line marks the 10 min horizon.\n"
        f"Right: shell starts {tw_r-t_out_r:.0f} K above outdoors, flushed 8 min, then either "
        f"shut or left open.    Dashed line in each panel: outdoor temperature.",
        gap=62)
    fig.savefig("results/figures/step04_window_fan.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("\n  wrote results/figures/step04_window_fan.png")

    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 4 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
