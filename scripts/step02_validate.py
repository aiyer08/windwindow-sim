"""
Step 2: check the room model against physics we can work out by hand.

A simulation is only worth trusting if it reproduces answers we already know.
Five independent checks:

  A  Energy is conserved. Heat in minus heat out equals the change in stored heat.
  B  The integrator is exact. The two-node system is linear for a fixed action,
     so it has a closed-form solution via the matrix exponential.
  C  Steady states match a hand-built resistance network (series and parallel
     conductances worked out separately from the simulator's own code).
  D  With the wall pinned, the room must follow Newton's law of cooling:
     T(t) = T_ss + (T_0 - T_ss) * exp(-t/tau), tau = C/K.
  E  The two time constants from the eigenvalues match the observed decay.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.linalg import expm

from windwindow import config as cfg
from windwindow.room import Room, linear_system, steady_state, k_in_wall, k_vent, fan_heat_w

TOL = {}
results = []

def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")

# --------------------------------------------------------------------------
print("\nA. Energy conservation")
# --------------------------------------------------------------------------
CASES_A = [(0, 0, 20, 17.6), (1, 1, 30, 17.6), (0.4, 0.6, 24, 8.8),
           (1, 0, 15, 0.0), (0.2, 1.0, 28, 17.6)]

def net_power(room, u, f, t_out):
    """Heat flowing in, minus heat flowing out to the world, in watts."""
    p_in = room.heater_w + fan_heat_w(f)
    p_out = ((k_vent(u, f) + cfg.K_LEAK) * (room.t_in - t_out)
             + cfg.K_WALL_OUT * (room.t_wall - t_out))
    return p_in - p_out

# A1: the instantaneous identity. C_in*dT_in/dt + C_wall*dT_wall/dt must equal
# the net power. This is what proves the two equations form one consistent
# energy balance -- that the air<->wall coupling appears with equal and
# opposite signs and no heat is invented at the internal surface.
worst = 0.0
for (u, f, t_out, heater) in CASES_A:
    room = Room(t_in=27.3, t_wall=31.1, heater_w=heater)
    di, dw = room.derivatives(room.t_in, room.t_wall, u, f, t_out)
    stored_rate = cfg.C_IN * di + cfg.C_WALL * dw
    worst = max(worst, abs(stored_rate - net_power(room, u, f, t_out)))
check("instantaneous energy balance", worst < 1e-9, f"worst error {worst:.2e} W")

# A2: the same thing integrated over a full hour of simulation. Total joules
# in minus joules out must equal the change in stored joules. The leftover
# mismatch here is the trapezoid rule's own error, not a physics error, so the
# check is that it shrinks like dt^2 when the step is made four times smaller.
def integrated_error(u, f, t_out, heater, dt, seconds=3600):
    room = Room(t_in=27.3, t_wall=31.1, heater_w=heater)
    e0 = cfg.C_IN * room.t_in + cfg.C_WALL * room.t_wall
    prev = net_power(room, u, f, t_out)
    joules = 0.0
    for _ in range(int(seconds / dt)):
        room.step(u, f, t_out, dt=dt)
        cur = net_power(room, u, f, t_out)
        joules += 0.5 * (prev + cur) * dt      # trapezoid rule
        prev = cur
    e1 = cfg.C_IN * room.t_in + cfg.C_WALL * room.t_wall
    scale = max(abs(joules), abs(e1 - e0), 1.0)
    return abs(joules - (e1 - e0)) / scale

coarse = max(integrated_error(*c, dt=1.0) for c in CASES_A)
fine = max(integrated_error(*c, dt=0.25) for c in CASES_A)
ratio = coarse / max(fine, 1e-18)
check("1 h integrated energy balance", fine < 3e-6 and 10.0 < ratio < 22.0,
      f"rel. error {coarse:.2e} at dt=1 s -> {fine:.2e} at dt=0.25 s "
      f"({ratio:.1f}x, dt^2 predicts 16x)")

# --------------------------------------------------------------------------
print("\nB. Integrator vs exact matrix-exponential solution")
# --------------------------------------------------------------------------
worst = 0.0
for (u, f, t_out, heater) in [(0, 0, 20, 17.6), (1, 1, 18, 17.6), (0.5, 0.5, 31, 17.6),
                              (1, 0, 26, 0.0), (0.3, 0.9, 22, 8.8)]:
    a, b = linear_system(u, f, t_out, heater)
    x0 = np.array([29.0, 33.0])
    for minutes in (1, 10, 60, 240):
        t = minutes * 60.0
        # Exact: x(t) = expm(A t) (x0 - x_ss) + x_ss, with x_ss = -A^-1 b
        x_ss = np.linalg.solve(a, -b)
        exact = expm(a * t) @ (x0 - x_ss) + x_ss
        room = Room(t_in=x0[0], t_wall=x0[1], heater_w=heater)
        for _ in range(int(t / cfg.DT_PHYSICS)):
            room.step(u, f, t_out)
        err = max(abs(room.t_in - exact[0]), abs(room.t_wall - exact[1]))
        worst = max(worst, err)
check("RK4 vs expm, 1-240 min", worst < 1e-8, f"worst error {worst:.2e} K")

# --------------------------------------------------------------------------
print("\nC. Steady state vs a hand-built resistance network")
# --------------------------------------------------------------------------
def hand_steady_state(u, f, t_out, heater):
    """Worked out from scratch, not using room.linear_system.

    Two paths carry heat from the inside air to outdoors:
      path 1  air -> wall -> outdoors   (two conductances in series)
      path 2  air -> outdoors directly  (the window opening plus the leak)
    In parallel, so their conductances add. Then dT = power / total.
    """
    mix = min(1.0, u * (cfg.MIX_WINDOW + (1 - cfg.MIX_WINDOW) * f))
    h = cfg.H_IN + (cfg.H_IN_FORCED - cfg.H_IN) * mix
    # Half the envelope resistance sits between the node and the inner surface.
    kiw = cfg.AREA_WALL / (1 / h + cfg.R_ENVELOPE / 2)
    kv = u * (cfg.K_VENT_OPEN_MAX + f * cfg.K_VENT_FAN_MAX) + cfg.K_LEAK
    k_series = 1.0 / (1.0 / kiw + 1.0 / cfg.K_WALL_OUT)   # air -> structure -> out
    k_total = k_series + kv                                # parallel
    power = heater + cfg.FAN_HEAT_FRAC * cfg.FAN_ELEC_W * f
    t_in = t_out + power / k_total
    # The wall sits partway along path 1, at the voltage-divider point.
    q_path1 = k_series * (t_in - t_out)
    t_wall = t_out + q_path1 / cfg.K_WALL_OUT
    return t_in, t_wall

worst = 0.0
for (u, f, t_out, heater) in [(0, 0, 20, 17.6), (1, 1, 20, 17.6), (1, 0, 20, 17.6),
                              (0.5, 0.5, 30, 8.8), (0.15, 0.3, 24, 17.6), (0, 0, 31, 0.0)]:
    hand = np.array(hand_steady_state(u, f, t_out, heater))
    code = steady_state(u, f, t_out, heater)
    # And confirm the simulator actually converges there. The real room has a
    # slow mode of about 12 hours, so settling needs on the order of 150 hours
    # rather than the 12 that sufficed for the small box.
    room = Room(t_in=t_out, t_wall=t_out, heater_w=heater)
    for _ in range(int(150 * 3600 / cfg.DT_PHYSICS)):
        room.step(u, f, t_out)
    sim = np.array([room.t_in, room.t_wall])
    worst = max(worst, np.max(np.abs(hand - code)), np.max(np.abs(hand - sim)))
check("hand network vs code vs 150 h sim", worst < 1e-5, f"worst error {worst:.2e} K")

# --------------------------------------------------------------------------
print("\nD. Newton's law of cooling, wall pinned")
# --------------------------------------------------------------------------
# Make the wall's heat store enormous so its temperature is effectively frozen.
# The inside air then has to obey the single-node textbook exponential.
real_c_wall = cfg.C_WALL
cfg.C_WALL = 1e12
worst = 0.0
for (u, f, t_out, heater, t_wall0, t_in0) in [
        (0, 0, 20, 17.6, 22.0, 30.0), (1, 1, 19, 17.6, 32.0, 24.0),
        (1, 0, 25, 0.0, 29.0, 21.0)]:
    kiw = k_in_wall(u, f)
    kv = k_vent(u, f) + cfg.K_LEAK
    k = kiw + kv
    tau = cfg.C_IN / k
    t_ss = (heater + fan_heat_w(f) + kiw * t_wall0 + kv * t_out) / k
    room = Room(t_in=t_in0, t_wall=t_wall0, heater_w=heater)
    for i in range(1, int(30 * 60 / cfg.DT_PHYSICS) + 1):
        room.step(u, f, t_out)
        t = i * cfg.DT_PHYSICS
        analytic = t_ss + (t_in0 - t_ss) * np.exp(-t / tau)
        worst = max(worst, abs(room.t_in - analytic))
    print(f"        u={u} f={f}: tau={tau:6.1f} s, T_ss={t_ss:6.3f} degC")
cfg.C_WALL = real_c_wall
check("T(t)=T_ss+(T0-T_ss)e^(-t/tau)", worst < 1e-6, f"worst error {worst:.2e} K")

# --------------------------------------------------------------------------
print("\nE. Time constants from the eigenvalues")
# --------------------------------------------------------------------------
for (u, f, label) in [(0, 0, "window shut"), (1, 0, "window open, fan off"),
                      (1, 1, "window open, fan on")]:
    a, _ = linear_system(u, f, 20.0, cfg.HEATER_W)
    ev = np.linalg.eigvals(a)
    taus = sorted(-1.0 / ev.real)
    print(f"        {label:22s} fast {taus[0]/60:5.2f} min   slow {taus[1]/60:6.2f} min")

a, _ = linear_system(0, 0, 20.0, cfg.HEATER_W)
taus = sorted(-1.0 / np.linalg.eigvals(a).real)
# Fit the slow decay of the closed room and compare with the slow eigenvalue.
ss = steady_state(0, 0, 20.0, cfg.HEATER_W)
room = Room(t_in=20.0, t_wall=20.0)
ts, dev = [], []
# Fit the slow decay well after the fast mode (about 68 min) has died away,
# over a span comparable with the slow mode itself (about 12 hours).
n_steps = int(220 * 3600 / cfg.DT_PHYSICS)
for i in range(1, n_steps + 1):
    room.step(0, 0, 20.0)
    t_now = i * cfg.DT_PHYSICS
    if t_now % 600 == 0 and t_now > 40 * 3600:
        ts.append(t_now); dev.append(abs(room.t_in - ss[0]))
slope = np.polyfit(ts, np.log(dev), 1)[0]
tau_fit = -1.0 / slope
err = abs(tau_fit - taus[1]) / taus[1]
check("fitted slow tau vs eigenvalue", err < 0.01,
      f"fit {tau_fit/60:.2f} min vs {taus[1]/60:.2f} min ({err*100:.2f}%)")

# --------------------------------------------------------------------------
ok = all(r[1] for r in results)
print(f"\nSTEP 2 DONE-WHEN: {'PASS' if ok else 'FAIL'}  ({sum(r[1] for r in results)}/{len(results)} checks)")
raise SystemExit(0 if ok else 1)
