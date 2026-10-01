"""
How good could any model get?

Before deciding whether a prediction error is good or bad, it helps to know
what the best possible error is. Here that can be computed exactly, because we
wrote the physics ourselves.

Three oracles, each knowing more than the last:

  perfect    the real physics, the real state, and the real future outdoor
             temperature. Only sensor noise in the label is left, so this is
             the floor set by the measurement itself.
  hold       the real physics and the real state, but the outdoor temperature
             assumed to stay where it is. This is the fair floor for our
             models, because the nine inputs contain no weather forecast.
  trend      the same, but with the outdoor temperature extrapolated along its
             5-minute trend.

The gap between `perfect` and `hold` is the cost of not knowing what the
weather will do next. No model built from the nine inputs can recover it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .room import Room, rollout_exact
from .weather import make_scenario


def oracle_predictions(df: pd.DataFrame, duration_s: float = 4 * 3600.0) -> pd.DataFrame:
    """Exact 10-minute predictions for every row, under the three oracles."""
    steps = int(round(cfg.HORIZON_S / cfg.DT_PHYSICS))
    out = {"perfect": [], "hold": [], "trend": []}

    for _, g in df.groupby("session_id", sort=False):
        heater = float(g["heater_w"].iloc[0])
        scen = make_scenario(str(g["scenario"].iloc[0]), seed=int(g["seed"].iloc[0]),
                             duration_s=duration_s, heater_w=heater)
        for _, r in g.iterrows():
            t0, u, f = float(r["t_s"]), float(r["u"]), float(r["f"])
            ti, tw = float(r["t_in_true"]), float(r["t_wall_true"])

            room = Room(t_in=ti, t_wall=tw, heater_w=heater)
            for k in range(steps):
                room.step(u, f, scen.t_out(t0 + k * cfg.DT_PHYSICS))
            out["perfect"].append(room.t_in - ti)

            t_out_now = float(r["t_out_true"])
            out["hold"].append(
                rollout_exact(ti, tw, u, f, t_out_now, heater, cfg.HORIZON_S)[0] - ti)

            slope_s = float(r["slope_out"]) / 60.0
            room = Room(t_in=ti, t_wall=tw, heater_w=heater)
            for k in range(steps):
                room.step(u, f, t_out_now + slope_s * k * cfg.DT_PHYSICS)
            out["trend"].append(room.t_in - ti)

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------
# The best any controller could do
# --------------------------------------------------------------------------
from .policies import BasePolicy, FixedActionPolicy, TimerPolicy   # noqa: E402


class GreedyOraclePolicy(BasePolicy):
    """A controller that cheats, to establish what is achievable.

    At every tick it uses the real physics, the room's true state (not the
    sensor readings) and the real outdoor temperature, and picks whichever
    action leaves the room coolest ten minutes later. No real controller could
    do this. Its purpose is to answer "how much of the gap to the timer was
    ever available to close?" before we judge how much of it our model closed.

    It is greedy, not globally optimal -- it optimises the next ten minutes,
    not the whole session. In this room that turns out not to matter: it lands
    within a degree-minute of simply ventilating flat out.
    """

    name = "oracle"
    CANDIDATES = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 0.0), (1.0, 1.0))

    def __init__(self, scenario):
        super().__init__(min_dwell_s=0.0)
        self.scenario = scenario

    def act(self, row: dict, history: list):
        t_in, t_wall = row["t_in_true"], row["t_wall_true"]
        t_out = self.scenario.t_out(row["t_s"])
        best_end, best_action = None, (0.0, 0.0)
        for u, f in self.CANDIDATES:
            end = rollout_exact(t_in, t_wall, u, f, t_out,
                                self.scenario.heater_w, cfg.HORIZON_S)[0]
            if best_end is None or end < best_end:
                best_end, best_action = end, (u, f)
        return self._set(row["t_s"], *best_action)


def comfort_bounds(scenario, sensor_seed: int) -> dict:
    """Degree-minutes above 26 degC for the reference policies and the bounds."""
    from .simulator import run_session, summarise

    out = {}
    for label, pol in (
        ("timer", TimerPolicy()),
        ("always_shut", FixedActionPolicy(0.0, 0.0)),
        ("always_open", FixedActionPolicy(1.0, 0.0)),
        ("always_open_fan", FixedActionPolicy(1.0, 1.0)),
        ("greedy_oracle", GreedyOraclePolicy(scenario)),
    ):
        s = summarise(run_session(scenario, pol, sensor_seed=sensor_seed))
        out[label] = s["degree_minutes_over_26"]
        out[label + "_fan_min"] = s["fan_minutes"]

    # The unreachable floor: what you would get if the room air simply matched
    # the outdoor air. Impossible here, because the heater always holds the
    # room some way above it, but it bounds everything.
    g = scenario.t_out_grid[::int(cfg.DT_CONTROL / cfg.DT_PHYSICS)]
    out["outdoor_floor"] = float(
        np.maximum(0.0, g - cfg.COMFORT_C).sum() * cfg.DT_CONTROL / 60.0)
    return out
