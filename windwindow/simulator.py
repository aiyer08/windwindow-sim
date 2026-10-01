"""
Step 6: the simulator loop and the data logger.

The referee. It runs every policy through exactly the same procedure so the
comparison in Step 12 is fair:

  * physics advances every 1 s
  * the sensors are polled every 5 s
  * every 30 s one row is written, holding the *average* of the readings taken
    during that window, and the policy is asked what to do next

Averaging over the window is what a real data logger does, and it matters: it
cuts the random noise by sqrt(6) while leaving the per-session calibration
offset completely untouched. The offset is the part the controller cannot
average away, which is why it is the part that can actually mislead it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .room import Room
from .sensors import SensorSuite
from .weather import Scenario

LOG_COLUMNS = [
    "t_s", "scenario", "seed", "policy", "session_id",
    "u", "f", "heater_w",
    "t_in", "t_wall", "t_out",                      # measured, 30 s averages
    "t_in_true", "t_wall_true", "t_out_true",       # truth, 30 s averages
    "fan_wh", "heater_wh", "fan_on_s", "above_comfort_min",
]


def run_session(
    scenario: Scenario,
    policy,
    sensor_seed: int | None = None,
    session_id: str | None = None,
) -> pd.DataFrame:
    """Run one policy through one weather scenario. Returns the logged rows.

    The policy sees only the measured, averaged values -- never the truth.
    """
    if sensor_seed is None:
        sensor_seed = 10_000 + scenario.seed
    if session_id is None:
        session_id = f"{scenario.name}_s{scenario.seed}_{policy.name}"

    room = Room(t_in=scenario.t_in0, t_wall=scenario.t_wall0, heater_w=scenario.heater_w)
    suite = SensorSuite(seed=sensor_seed)
    if hasattr(policy, "reset"):
        policy.reset(scenario)

    # Heat gains vary through the day in a real room. A scenario that provides
    # gain_w() is asked for the value at each step; the legacy box scenarios do
    # not and fall back to their constant load.
    gain_at = getattr(scenario, "gain_w", None)
    if gain_at is None:
        _const_gain = float(getattr(scenario, "heater_w", cfg.HEATER_W))
        def gain_at(_t, _g=_const_gain):
            return _g

    steps_per_tick = int(round(cfg.DT_CONTROL / cfg.DT_PHYSICS))
    steps_per_read = int(round(cfg.DT_SENSOR / cfg.DT_PHYSICS))
    n_ticks = int(scenario.duration_s / cfg.DT_CONTROL)

    rows: list[dict] = []
    u, f = 0.0, 0.0

    for tick in range(n_ticks):
        t0 = tick * cfg.DT_CONTROL

        # Ask the policy what to do for the coming interval. On the first tick
        # it has no history yet, so it starts from the window shut.
        if rows:
            u, f = policy.act(rows[-1], rows)

        acc_meas = {"t_in": [], "t_wall": [], "t_out": []}
        acc_true = {"t_in": [], "t_wall": [], "t_out": []}
        acc_gain = []
        fan_on_s = 0.0
        wh0, hwh0 = room.fan_wh, room.heater_wh

        for i in range(steps_per_tick):
            t = t0 + i * cfg.DT_PHYSICS
            t_out = scenario.t_out(t)
            gain = gain_at(t)
            suite.update(room.t_in, room.t_wall, t_out, cfg.DT_PHYSICS)
            if i % steps_per_read == 0:
                m = suite.read()
                for c in acc_meas:
                    acc_meas[c].append(m[c])
                acc_true["t_in"].append(room.t_in)
                acc_true["t_wall"].append(room.t_wall)
                acc_true["t_out"].append(t_out)
                acc_gain.append(gain)
            if f > 0.05:
                fan_on_s += cfg.DT_PHYSICS
            room.step(u, f, t_out, gain_w=gain)

        t_in_true = float(np.mean(acc_true["t_in"]))
        rows.append({
            "t_s": t0,
            "scenario": scenario.name,
            "seed": scenario.seed,
            "policy": policy.name,
            "session_id": session_id,
            "u": u,
            "f": f,
            # What the controller is allowed to know about the current gain.
            "heater_w": float(np.mean(acc_gain)),
            "t_in": float(np.mean(acc_meas["t_in"])),
            "t_wall": float(np.mean(acc_meas["t_wall"])),
            "t_out": float(np.mean(acc_meas["t_out"])),
            "t_in_true": t_in_true,
            "t_wall_true": float(np.mean(acc_true["t_wall"])),
            "t_out_true": float(np.mean(acc_true["t_out"])),
            "fan_wh": room.fan_wh - wh0,
            "heater_wh": room.heater_wh - hwh0,
            "fan_on_s": fan_on_s,
            "above_comfort_min": max(0.0, t_in_true - cfg.COMFORT_C) * cfg.DT_CONTROL / 60.0,
        })

    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def summarise(df: pd.DataFrame) -> dict:
    """The headline numbers for one session -- the inputs to goals G3 and G4."""
    minutes = len(df) * cfg.DT_CONTROL / 60.0
    return {
        "scenario": df["scenario"].iloc[0],
        "seed": int(df["seed"].iloc[0]),
        "policy": df["policy"].iloc[0],
        "session_id": df["session_id"].iloc[0],
        "minutes": minutes,
        "mean_t_in": df["t_in_true"].mean(),
        "max_t_in": df["t_in_true"].max(),
        "degree_minutes_over_26": df["above_comfort_min"].sum(),
        "minutes_over_26": float((df["t_in_true"] > cfg.COMFORT_C).sum() * cfg.DT_CONTROL / 60.0),
        "minutes_under_24": float((df["t_in_true"] < cfg.T_IN_FLOOR_C).sum() * cfg.DT_CONTROL / 60.0),
        "fan_minutes": df["fan_on_s"].sum() / 60.0,
        "fan_wh": df["fan_wh"].sum(),
        "window_open_minutes": float((df["u"] > 0.05).sum() * cfg.DT_CONTROL / 60.0),
        "moves": int((df["u"].diff().abs() > 0.05).sum()),
    }
