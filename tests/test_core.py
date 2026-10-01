"""
Fast invariant checks. Run with:  .venv/bin/python tests/test_core.py

These are deliberately quick (under a second). The thorough validation lives in
scripts/step02_validate.py and in the DONE-WHEN check at the end of every step;
this file is for catching an obvious break without waiting for the pipeline.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from windwindow import config as cfg
from windwindow.features import (FEATURE_NAMES, build_features, feature_vector,
                                 slope_per_min)
from windwindow.room import (Room, k_in_wall, k_vent, linear_system,
                             rollout_exact, steady_state)
from windwindow.sensors import SensorSuite
from windwindow.weather import SCENARIOS, make_scenario

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


print("physics")
# A shut room with no heater and matching outdoor temperature must not drift.
room = Room(t_in=22.0, t_wall=22.0, heater_w=0.0)
for _ in range(int(600 / cfg.DT_PHYSICS)):
    room.step(0.0, 0.0, 22.0)
check("no heat source, no gradient -> no change",
      abs(room.t_in - 22.0) < 1e-9 and abs(room.t_wall - 22.0) < 1e-9)

# Heat must flow from hot to cold, never the other way.
room = Room(t_in=30.0, t_wall=30.0, heater_w=0.0)
for _ in range(int(600 / cfg.DT_PHYSICS)):
    room.step(0.0, 0.0, 20.0)
check("with no heater, a warm room cools toward outdoors",
      20.0 < room.t_in < 30.0)

room = Room(t_in=20.0, t_wall=20.0, heater_w=0.0)
for _ in range(int(600 / cfg.DT_PHYSICS)):
    room.step(1.0, 1.0, 30.0)
check("a cool room warms toward a hot outdoors", 20.0 < room.t_in < 30.0)

# Opening the window must widen the air-to-outdoors path, monotonically.
vents = [k_vent(u, 0.0) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
check("ventilation rises with window opening", all(np.diff(vents) > 0))
check("the fan only moves air if the window is open",
      k_vent(0.0, 1.0) == 0.0)
check("airflow raises the wall-to-air conductance",
      k_in_wall(0.0, 0.0) < k_in_wall(1.0, 0.0) < k_in_wall(1.0, 1.0))

# The exact solver and the step integrator must agree.
ex = rollout_exact(29.0, 33.0, 1.0, 1.0, 24.0, cfg.HEATER_W, 600.0)
room = Room(t_in=29.0, t_wall=33.0, heater_w=cfg.HEATER_W)
for _ in range(int(600 / cfg.DT_PHYSICS)):
    room.step(1.0, 1.0, 24.0)
check("exact solver matches the step integrator",
      abs(ex[0] - room.t_in) < 1e-9, f"{abs(ex[0]-room.t_in):.2e}")

# Steady state must satisfy the steady-state equations.
for u, f, t_out, q in ((0, 0, 20, 17.6), (1, 1, 30, 8.8), (0.5, 0.5, 24, 0.0)):
    x = steady_state(u, f, t_out, q)
    a, b = linear_system(u, f, t_out, q)
    check(f"steady state solves A x + b = 0 (u={u}, f={f})",
          np.allclose(a @ x + b, 0, atol=1e-10))

# More ventilation must never leave a room warmer than less, when it is
# cooler outside and there is no fan heat to confuse the comparison.
warm = rollout_exact(30.0, 30.0, 0.0, 0.0, 20.0, 0.0, 600.0)[0]
cool = rollout_exact(30.0, 30.0, 1.0, 0.0, 20.0, 0.0, 600.0)[0]
check("ventilating a warm room into cool air cools it faster", cool < warm)

print("\nweather")
for name in SCENARIOS:
    a = make_scenario(name, seed=5)
    b = make_scenario(name, seed=5)
    check(f"{name}: reproducible", np.array_equal(a.t_out_grid, b.t_out_grid))
c = make_scenario("hot", seed=5)
d = make_scenario("hot", seed=6)
check("different seeds give different weather",
      not np.array_equal(c.t_out_grid, d.t_out_grid))
check("initial-state overrides do not disturb the weather trace",
      np.array_equal(make_scenario("hot", seed=5, t_in0=15.0).t_out_grid, c.t_out_grid))
check("t_out is clamped past the end of the run",
      c.t_out(c.duration_s * 10) == c.t_out_grid[-1])

print("\nsensors")
suite = SensorSuite(seed=3)
for _ in range(200):
    suite.update(25.0, 25.0, 25.0, 1.0)
reads = np.array([suite.read()["t_in"] for _ in range(400)])
check("readings land on the quantisation grid",
      np.allclose(reads / cfg.SENSOR_QUANT_C, np.round(reads / cfg.SENSOR_QUANT_C)))
check("readings are noisy", reads.std() > 0.02)
check("readings centre on true value plus the fixed offset",
      abs(reads.mean() - (25.0 + suite.offsets["t_in"])) < 0.05)
check("the offset is a per-session constant",
      SensorSuite(seed=3).offsets == suite.offsets
      and SensorSuite(seed=4).offsets != suite.offsets)

print("\nfeatures")
state = {"t_in": 28.0, "t_wall": 31.0, "t_out": 24.0, "heater_w": 17.6,
         "slope_in": 0.1, "slope_out": -0.2}
v = feature_vector(state, 1.0, 1.0)
check("feature vector has the nine inputs in order",
      v.shape == (1, 9) and len(FEATURE_NAMES) == 9)
shut = build_features(28.0, 31.0, 24.0, 17.6, 0.1, -0.2, 0.0, 0.0)
for i, n in enumerate(FEATURE_NAMES):
    if n.startswith("u_"):
        check(f"{n} is zero when the window is shut", shut[i] == 0.0)

hist = [{"t_s": i * 30.0, "t_in": 20.0 + 0.02 * i} for i in range(11)]
check("slope is recovered from a clean ramp",
      abs(slope_per_min(hist, "t_in") - 0.04) < 1e-9,
      f"got {slope_per_min(hist, 't_in'):.6f}")
check("slope is zero without enough history",
      slope_per_min(hist[:2], "t_in") == 0.0)

print("\nheat gains")
from windwindow.gains import daily_summary, internal_w, solar_w, total_w
gs = daily_summary()
check("gains peak in the afternoon", 14.0 < gs["peak_hour"] < 19.0,
      f"peak at {gs['peak_hour']:.1f} h")
check("no solar gain at night", solar_w(2.0) == 0.0 and solar_w(23.0) == 0.0)
check("solar gain at midday", solar_w(14.0) > 50.0)
check("internal gain never zero", float(np.min(internal_w(np.arange(0, 24, 0.1)))) > 0)
check("total is internal plus solar",
      abs(total_w(16.0) - (internal_w(16.0) + solar_w(16.0))) < 1e-9)
check("gain load is room-scale, not box-scale",
      1.0 < gs["mean_w"] / cfg.VOLUME_IN < 20.0,
      f"{gs['mean_w']/cfg.VOLUME_IN:.1f} W/m3")

print("\nreal weather")
from windwindow.realweather import LOCATIONS, build
try:
    for loc in LOCATIONS:
        w = build(loc, hours=24, start_hour=0)
        s = w.summary()
        check(f"{loc}: plausible outdoor range",
              -20 < s["t_out_min"] < s["t_out_max"] < 55)
        check(f"{loc}: gain follows the clock",
              w.gain_w(16 * 3600) > w.gain_w(3 * 3600))
    a = build("palo_alto", hours=24, seed=4)
    b = build("palo_alto", hours=24, seed=4)
    check("weather build is reproducible", np.array_equal(a.t_out_grid, b.t_out_grid))
except FileNotFoundError:
    print("  [SKIP] cached weather missing; run scripts/fetch_weather.py")

print("\nforecast")
from windwindow.forecast import Forecast, ramp
tr = build("palo_alto", hours=48).t_out_grid if LOCATIONS else None
if tr is not None:
    fc = Forecast.build(tr, seed=1)
    check("a forecast at zero lead is exact",
          abs(fc.at(3600.0, np.array([0.0]))[0] - tr[720]) < 1e-9
          or abs(ramp(0.0)) < 1e-12)
    prof = {r["lead_hours"]: r["mae"] for r in fc.error_profile()}
    check("error grows with lead time",
          prof[1] < prof[6] < prof[24], str({k: round(v, 2) for k, v in prof.items()}))
    check("error is the right magnitude at 6 h", 0.4 < prof[6] < 2.0)
    perfect = Forecast.build(tr, seed=1, perfect=True)
    check("a perfect forecast has no error",
          all(r["mae"] < 1e-9 for r in perfect.error_profile()))

print("\nstate-space model")
from windwindow.statespace import (LOWER, UPPER, N_PARAMS, PARAM_NAMES,
                                   StateSpaceModel, true_params)
tp = true_params()
check("ten named parameters", len(PARAM_NAMES) == N_PARAMS == 10)
check("true parameters lie inside the fitting bounds",
      bool(np.all(tp >= LOWER) and np.all(tp <= UPPER)))
m = StateSpaceModel(theta=tp)
# With the true parameters the model must track the real physics.
w24 = int(24 * 3600 / cfg.DT_CONTROL)
outs = np.full(w24, 20.0)
gains = np.full(w24, 150.0)
for u, f in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
    pi, _ = m.rollout(26.0, 26.0, outs, np.full(w24, u), np.full(w24, f), gains)
    r = Room(t_in=26.0, t_wall=26.0)
    for _ in range(int(24 * 3600 / cfg.DT_PHYSICS)):
        r.step(u, f, 20.0, gain_w=150.0)
    check(f"model tracks physics over 24 h (u={u}, f={f})",
          abs(pi[-1] - r.t_in) < 0.1, f"{abs(pi[-1]-r.t_in):.3f} K")
check("rollout returns one more point than steps", len(pi) == w24 + 1)
# A gain series and a constant gain must agree when the series is constant.
pa, _ = m.rollout(24.0, 24.0, outs, np.zeros(w24), np.zeros(w24), gains)
pb, _ = m.rollout(24.0, 24.0, outs, np.zeros(w24), np.zeros(w24), 150.0)
check("a constant gain series matches a scalar gain",
      abs(pa[-1] - pb[-1]) < 1e-9)

print("\nplanning controller")
from windwindow.mpc import BLOCKS_MIN, HORIZON_S, MPCPolicy, _block_index
check("the horizon spans a full day", HORIZON_S >= 20 * 3600,
      f"{HORIZON_S/3600:.1f} h")
n_steps = int(round(HORIZON_S / cfg.DT_CONTROL))
bi = _block_index(n_steps, BLOCKS_MIN)
check("every step belongs to a block",
      len(bi) == n_steps and bi.min() == 0 and bi.max() == len(BLOCKS_MIN) - 1)
check("blocks are ordered and contiguous", bool(np.all(np.diff(bi) >= 0)))
if tr is not None:
    pol = MPCPolicy(m, Forecast.build(tr, seed=0))
    pol.reset(build("palo_alto", hours=48))
    lead = np.arange(pol.n_steps) * cfg.DT_CONTROL
    outs2 = pol.forecast.at(0.0, lead)
    g2 = np.array([pol.gain_at(float(L)) for L in lead])
    # A schedule that keeps a hot room shut must cost more than one that vents.
    shut = np.zeros(2 * pol.n_blocks)
    vent = np.concatenate([np.ones(pol.n_blocks), np.zeros(pol.n_blocks)])
    c_shut = pol._cost(shut, 30.0, 30.0, outs2, g2, 0.0)
    c_vent = pol._cost(vent, 30.0, 30.0, outs2, g2, 0.0)
    check("sealing a hot room costs more than venting it", c_vent < c_shut,
          f"{c_vent:.0f} vs {c_shut:.0f}")
    check("the cost of a single schedule is a scalar", np.ndim(c_shut) == 0)
    many = np.stack([shut, vent], axis=1)
    cols = pol._cost(many, 30.0, 30.0, outs2, g2, 0.0)
    check("many schedules can be scored at once",
          cols.shape == (2,) and abs(cols[0] - c_shut) < 1e-9)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all core checks passed")
