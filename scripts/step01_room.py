"""Step 1: does a sealed room with a constant heat gain warm smoothly and level off?"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.room import Room, steady_state

T_OUT = 20.0
# The real room has a slow mode of about 12 hours, so it needs roughly three
# days to settle. The original box managed it in six hours.
HOURS = 72.0

def main():
    room = Room(t_in=20.0, t_wall=20.0, heater_w=cfg.HEATER_W)
    n = int(HOURS * 3600 / cfg.DT_PHYSICS)
    t, ti, tw = [0.0], [room.t_in], [room.t_wall]
    for i in range(n):
        room.step(u=0.0, f=0.0, t_out=T_OUT)
        t.append((i + 1) * cfg.DT_PHYSICS / 3600.0)
        ti.append(room.t_in)
        tw.append(room.t_wall)
    t, ti, tw = np.array(t), np.array(ti), np.array(tw)

    ss = steady_state(0.0, 0.0, T_OUT, cfg.HEATER_W)
    print(f"steady state (solved):    T_in={ss[0]:.4f}  T_wall={ss[1]:.4f}")
    print(f"after {HOURS:.0f} h (simulated): T_in={ti[-1]:.4f}  T_wall={tw[-1]:.4f}")

    # "Smoothly": T_in must rise monotonically and its rate must never increase
    # (a pure exponential approach to a ceiling).
    d = np.diff(ti)
    monotonic = bool(np.all(d > -1e-12))
    # allow tiny numerical wiggle in the second difference
    decelerating = bool(np.all(np.diff(d) < 1e-9))
    settled = abs(ti[-1] - ss[0]) < 0.05
    overshoot = ti.max() - ss[0]

    # Time to get 63% of the way there (one time constant).
    frac = (ti - ti[0]) / (ss[0] - ti[0])
    tau_min = t[np.searchsorted(frac, 1 - 1 / np.e)]
    print(f"rise 20 -> {ss[0]:.2f} degC  ({ss[0]-20:.2f} K above outdoors)")
    print(f"time constant (63% of the rise): {tau_min:.1f} h")
    print(f"monotonic rise: {monotonic}   decelerating: {decelerating}")
    print(f"overshoot above steady state:   {overshoot:+.4f} K")
    print(f"within 0.05 K of steady state at {HOURS:.0f} h: {settled}")

    P.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.axhline(ss[0], color=P.AXIS, lw=0.5, ls=(0, (3, 3)), zorder=1)
    ax.annotate(f"levels off at {ss[0]:.1f} °C", (t[-1], ss[0]),
                textcoords="offset points", xytext=(-4, 7), ha="right",
                color=P.INK_MUTED, fontsize=7)
    ax.plot(t, ti, color=P.TEMP_COLOR["t_in"], label=P.TEMP_LABEL["t_in"], zorder=4)
    ax.plot(t, tw, color=P.TEMP_COLOR["t_wall"], label=P.TEMP_LABEL["t_wall"], zorder=3)
    ax.plot(t, np.full_like(t, T_OUT), color=P.TEMP_COLOR["t_out"],
            label=P.TEMP_LABEL["t_out"], zorder=2)
    for key, arr in (("t_in", ti), ("t_wall", tw)):
        P.direct_label(ax, t[-1], arr[-1], P.TEMP_LABEL[key], P.TEMP_COLOR[key])
    P.direct_label(ax, t[-1], T_OUT, P.TEMP_LABEL["t_out"], P.TEMP_COLOR["t_out"])
    ax.set_title(f"Sealed room, {cfg.HEATER_W:.0f} W constant gain, "
                 f"{T_OUT:.0f} °C outside")
    ax.set_xlabel("hours")
    ax.set_ylabel("°C")
    ax.set_xlim(0, t[-1] * 1.005)
    ax.margins(y=0.12)
    P.legend_below(ax, ncol=3)
    P.save(fig, "results/figures/step01_warmup.png", ax=ax, note_gap=62,
           note=f"{cfg.HEATER_W:.0f} W constant gain, window shut. Time "
                f"constant {tau_min:.1f} h; settles {ss[0]-T_OUT:.1f} K above "
                f"outdoors.")

    ok = monotonic and decelerating and settled and overshoot < 0.01
    print("\nSTEP 1 DONE-WHEN:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
