"""Step 5: do the sensor readings wobble around the true values?"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.room import Room
from windwindow.sensors import SensorSuite, CHANNELS
from windwindow.weather import make_scenario

def main():
    sc = make_scenario("hot_then_cooling", seed=2, duration_s=90 * 60)
    room = Room(t_in=sc.t_in0, t_wall=sc.t_wall0, heater_w=sc.heater_w)
    suite = SensorSuite(seed=11)
    print("per-session calibration offsets (degC):",
          {k: round(v, 3) for k, v in suite.offsets.items()})

    rows = []
    n = int(sc.duration_s / cfg.DT_PHYSICS)
    tick = int(cfg.DT_CONTROL / cfg.DT_PHYSICS)
    # Toggle the window every 20 min so there are sharp edges for the lag to blunt.
    for i in range(n):
        t = i * cfg.DT_PHYSICS
        u = 1.0 if (int(t // (20 * 60)) % 2 == 1) else 0.0
        t_out = sc.t_out(t)
        suite.update(room.t_in, room.t_wall, t_out, cfg.DT_PHYSICS)
        if i % tick == 0:
            m = suite.read()
            rows.append((t, room.t_in, room.t_wall, t_out,
                         m["t_in"], m["t_wall"], m["t_out"], u))
        room.step(u, 0.0, t_out)

    a = np.array(rows)
    t = a[:, 0] / 60.0
    true = {"t_in": a[:, 1], "t_wall": a[:, 2], "t_out": a[:, 3]}
    meas = {"t_in": a[:, 4], "t_wall": a[:, 5], "t_out": a[:, 6]}
    u_hist = a[:, 7]

    print(f"\n{'channel':9s} {'bias':>7s} {'std of error':>13s} {'max |error|':>12s} "
          f"{'unique steps':>13s}")
    checks = []
    for c in CHANNELS:
        err = meas[c] - true[c]
        # Measured values must land on the quantisation grid.
        on_grid = np.allclose(meas[c] / cfg.SENSOR_QUANT_C,
                              np.round(meas[c] / cfg.SENSOR_QUANT_C), atol=1e-9)
        print(f"{c:9s} {err.mean():+7.3f} {err.std():13.3f} {np.abs(err).max():12.3f} "
              f"{len(np.unique(meas[c])):13d}")
        checks.append((f"{c}: readings wobble but track the truth",
                       0.02 < err.std() < 0.6 and np.abs(err).max() < 2.0 and on_grid))
        checks.append((f"{c}: readings sit on the 0.0625 degC grid", on_grid))

    # The bias must persist, not average away -- that is the point of an offset.
    halves = [np.mean(meas["t_in"][:len(t)//2] - true["t_in"][:len(t)//2]),
              np.mean(meas["t_in"][len(t)//2:] - true["t_in"][len(t)//2:])]
    print(f"\nT_in bias in the first half {halves[0]:+.3f}, second half {halves[1]:+.3f} "
          f"-- a fixed offset does not average away")
    checks.append(("the offset persists across the session",
                   abs(halves[0] - halves[1]) < 0.25))

    # And the lag must be visible: the reading should trail the truth after a
    # step change. Which way it trails depends on which way the room moves,
    # and that depends on whether the outdoor air is warmer or cooler, so the
    # check compares signs rather than assuming the room cools.
    edge = int(20 * 60 / cfg.DT_CONTROL)
    span = max(2, int(10 * 60 / cfg.DT_CONTROL))
    d_true = true["t_in"][edge + span] - true["t_in"][edge]
    lag_err = (meas["t_in"][edge:edge + span] - true["t_in"][edge:edge + span]).mean()
    rate = d_true / (span * cfg.DT_CONTROL / 60.0)
    direction = "rising" if d_true > 0 else "falling"
    print(f"right after the window opens the room is {direction} at "
          f"{rate:+.3f} degC/min, and the logged")
    print(f"reading trails it by {lag_err:+.3f} degC: the right sign, but "
          f"almost nothing.")

    # Why so little: the lag is real but the logging interval hides it. Each
    # logged row is a 5-minute average of readings taken every 15 s, and
    # averaging the reading and the truth over the same window removes most of
    # a 60 s lag. Demonstrate the lag directly on the sensor element, at the
    # sampling rate, so the model is shown to be doing what it claims.
    from windwindow.sensors import Sensor
    probe = Sensor(offset=0.0, noise_c=0.0)
    ramp_rate = 0.05          # degC per second, a deliberately fast change
    truth_now = 20.0
    for _ in range(int(600 / cfg.DT_SENSOR)):
        truth_now += ramp_rate * cfg.DT_SENSOR
        probe.update(truth_now, cfg.DT_SENSOR)
    raw_lag = probe.state - truth_now
    # On a steady ramp a first-order element settles to a lag of tau times the
    # rate. Updating in discrete steps, with the truth advanced before the
    # element chases it, removes half a sample step from that.
    expected = -(cfg.SENSOR_TAU_S - cfg.DT_SENSOR / 2.0) * ramp_rate
    print(f"on a {ramp_rate:.2f} degC/s ramp the element itself trails by "
          f"{raw_lag:+.2f} degC,")
    print(f"against {expected:+.2f} degC predicted for a "
          f"{cfg.SENSOR_TAU_S:.0f} s time constant sampled every "
          f"{cfg.DT_SENSOR:.0f} s")

    checks.append(("the logged reading trails in the right direction",
                   bool((lag_err * d_true) < 0)))
    checks.append(("the lag is negligible once averaged over the logging interval",
                   abs(lag_err) < 0.05))
    checks.append(("the element itself lags by tau times the rate of change",
                   abs(raw_lag - expected) < 0.02 * abs(expected)))

    # ------------------------------------------------------------------ figure
    P.use_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    for c in CHANNELS:
        ax.plot(t, true[c], color=P.TEMP_COLOR[c], lw=1.0, zorder=3)
        ax.plot(t, meas[c], color=P.TEMP_COLOR[c], lw=0.0, marker="o", ms=1.7,
                alpha=0.75, zorder=4, label=f"{P.TEMP_LABEL[c]} (measured)")
        P.direct_label(ax, t[-1], true[c][-1], P.TEMP_LABEL[c], P.TEMP_COLOR[c], dx=5)
    ax.set_ylabel("°C")
    ax.set_title("Step 5  ·  Sensor readings wobble around the true values")
    ax.set_xlim(0, t[-1] * 1.14)

    ax2 = axes[1]
    ax2.fill_between(t, 0, u_hist, step="post", color=P.GRID, zorder=2)
    ax2.plot(t, u_hist, drawstyle="steps-post", color=P.INK_MUTED, lw=0.8, zorder=3)
    ax2.set_ylim(-0.1, 1.35)
    ax2.set_yticks([0, 1], ["shut", "open"])
    ax2.set_ylabel("window")
    ax2.set_xlabel("minutes")
    ax2.grid(False)

    P.legend_below(axes[1], ncol=3, gap=32, source=axes[0])
    note = P.caption(axes[1],
                     "Lines: the true temperatures. Dots: what the sensors report every 30 s, "
                     "with a per-session offset, a 20 s lag, 0.1 °C noise and "
                     "0.0625 °C steps.", gap=52)
    fig.tight_layout()
    fig.savefig("results/figures/step05_sensors.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note])
    plt.close(fig)
    print("  wrote results/figures/step05_sensors.png")

    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 5 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
