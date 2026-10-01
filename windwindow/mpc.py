"""
The planning controller.

The greedy controller in policies.ModelPolicy compares two actions over the
next ten minutes and takes whichever looks better. That's the right thing to
do if nothing past ten minutes matters. In a room whose structure holds
1.5 MJ/K it isn't: heat you put into the structure at nine in the morning is
still there at three in the afternoon, and cold you pull out of it overnight is
room to absorb the afternoon. A ten-minute comparison can't see either,
because the payoff shows up hours after the decision.

So this one plans. Every so often it:

  1. reads the current state off the sensors,
  2. takes a forecast of outdoor temperature over the horizon,
  3. searches over schedules of window and fan settings,
  4. scores each one by rolling the identified model forward,
  5. does the first block of the best schedule and bins the rest.

Binning the rest is the point. The forecast six hours out is poor, so
committing to it would be silly, but it's good enough to tell you whether to
ventilate right now.

The schedule uses move blocking: short blocks near the present, where the
controller really has authority, and long ones further out, where the plan only
needs to capture rough intent. Keeps the search down to 24 numbers.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from .policies import BasePolicy

# Block lengths in minutes: short near the present, where control authority is
# real, and long further out, where the plan only needs to capture intent.
#
# The horizon has to span a full day. A controller deciding at 2 am whether to
# flush the room needs to see the following afternoon, because that is when the
# benefit of having cooled the structure arrives. A 7.5 hour horizon was tried
# first and lost to the threshold rule for exactly this reason: at 2 am it
# could see no further than half past nine, so pre-cooling looked like
# pointless overcooling and the planner declined to do it.
BLOCKS_MIN = (15, 15, 30, 30, 60, 60, 120, 120, 180, 240, 240, 300)
HORIZON_S = sum(BLOCKS_MIN) * 60.0          # 23.5 hours
# Re-plan at the same interval as the first block, so the block boundaries
# stay aligned with the clock as the horizon slides forward.
REPLAN_S = 15 * 60.0
PLAN_DT = cfg.DT_CONTROL                    # integrate at the identified step

# Objective weights.
FLOOR_WEIGHT = 1.0        # cost of being below the floor, per degree-minute
MOVE_WEIGHT = 1.5         # cost of changing the window position, per unit step
# Degree-minutes charged per watt-hour of fan electricity. The fan buys about
# 0.6 degree-minutes per watt-hour in this room, so this is the value that
# decides whether it runs at all. Swept in step 18.
DEFAULT_FAN_PENALTY = 0.35


def _block_index(n_steps: int, blocks_min=BLOCKS_MIN) -> np.ndarray:
    """Which block each integration step belongs to."""
    idx = np.empty(n_steps, dtype=int)
    pos = 0
    for b, mins in enumerate(blocks_min):
        k = int(round(mins * 60.0 / PLAN_DT))
        end = min(pos + k, n_steps)
        idx[pos:end] = b
        pos = end
        if pos >= n_steps:
            break
    idx[pos:] = len(blocks_min) - 1
    return idx


class MPCPolicy(BasePolicy):
    """Receding-horizon controller over continuous window and fan settings."""

    name = "mpc"

    def __init__(self, model, forecast, fan_penalty=DEFAULT_FAN_PENALTY,
                 comfort=cfg.COMFORT_C, floor=cfg.T_IN_FLOOR_C,
                 blocks_min=BLOCKS_MIN, replan_s=REPLAN_S, max_iter=25,
                 record=False, name=None, use_forecast=True):
        super().__init__(min_dwell_s=0.0)      # movement is penalised, not banned
        self.model = model
        self.forecast = forecast
        self.fan_penalty = float(fan_penalty)
        self.comfort = float(comfort)
        self.floor = float(floor)
        self.blocks_min = tuple(blocks_min)
        self.replan_s = float(replan_s)
        self.max_iter = int(max_iter)
        self.use_forecast = bool(use_forecast)
        self.record = record
        self.n_blocks = len(self.blocks_min)
        self.horizon_s = sum(self.blocks_min) * 60.0
        self.n_steps = int(round(self.horizon_s / PLAN_DT))
        self.bidx = _block_index(self.n_steps, self.blocks_min)
        self.trace: list = []
        self._plan = None
        self._next_replan = -1e9
        self.solves = 0
        self.gain_at = lambda _t: cfg.GAIN_BASE_W

    # ------------------------------------------------------------------ setup
    def reset(self, scenario=None) -> None:
        super().reset(scenario)
        self._plan = None
        self._next_replan = -1e9
        self.trace = []
        self.solves = 0
        if scenario is not None:
            g = getattr(scenario, "gain_w", None)
            if g is not None:
                self.gain_at = g
            else:
                const = float(getattr(scenario, "heater_w", cfg.GAIN_BASE_W))
                self.gain_at = lambda _t, _c=const: _c

    # -------------------------------------------------------------- objective
    def _cost(self, params, t_in, t_wall, t_out_series, gain_series, u_now):
        """Score one or many candidate schedules.

        params has shape (2 * n_blocks,) or (2 * n_blocks, n_candidates), so
        the finite-difference gradient evaluates every perturbation in one
        vectorised pass instead of looping.
        """
        p = np.atleast_2d(params.T).T if params.ndim == 1 else params
        single = params.ndim == 1
        p = params.reshape(2 * self.n_blocks, -1)
        u_b = np.clip(p[:self.n_blocks], 0.0, 1.0)
        f_b = np.clip(p[self.n_blocks:], 0.0, 1.0)
        ncand = p.shape[1]

        ti = np.full(ncand, t_in, dtype=float)
        tw = np.full(ncand, t_wall, dtype=float)
        over = np.zeros(ncand)
        under = np.zeros(ncand)
        fan_wh = np.zeros(ncand)
        dt_min = PLAN_DT / 60.0

        for k in range(self.n_steps):
            b = self.bidx[k]
            u = u_b[b]
            f = f_b[b]
            ti, tw = self.model.step(ti, tw, t_out_series[k], u, f,
                                     gain_series[k], PLAN_DT)
            over += np.maximum(0.0, ti - self.comfort) * dt_min
            under += np.maximum(0.0, self.floor - ti) * dt_min
            fan_wh += f * cfg.FAN_ELEC_W * PLAN_DT / 3600.0

        # Penalise window movement, including the step away from the current
        # position, so the controller does not fidget.
        moves = np.abs(np.diff(np.vstack([np.full(ncand, u_now), u_b]), axis=0))
        move_cost = MOVE_WEIGHT * moves.sum(axis=0)

        total = (over + FLOOR_WEIGHT * under
                 + self.fan_penalty * fan_wh + move_cost)
        return total[0] if single else total

    def _solve(self, t_in, t_wall, t_out_series, gain_series, u_now):
        """Minimise the cost over the schedule, starting from the last plan."""
        n = 2 * self.n_blocks

        def obj(x):
            return float(self._cost(np.asarray(x, dtype=float), t_in, t_wall,
                                    t_out_series, gain_series, u_now))

        def grad(x):
            x = np.asarray(x, dtype=float)
            eps = 1e-3
            # One column per perturbed coordinate, plus the base point.
            cols = np.repeat(x[:, None], n + 1, axis=1)
            cols[np.arange(n), np.arange(n)] += eps
            vals = self._cost(cols, t_in, t_wall, t_out_series, gain_series, u_now)
            return (vals[:n] - vals[n]) / eps

        from scipy.optimize import minimize

        # Always try several starting schedules and keep the best result.
        #
        # Warm-starting from the previous plan alone is not enough. The blocks
        # are uneven, so the schedule cannot simply be shifted along without
        # drifting out of step with the clock, and an objective with kinks at
        # the comfort threshold gives a local optimiser plenty of places to
        # stall. An earlier version warm-started only, and held the window
        # fully open for two days straight while its own cost function scored
        # that schedule worse than the one it had already found.
        always_open = np.concatenate([np.ones(self.n_blocks),
                                      np.zeros(self.n_blocks)])
        starts = [always_open, np.zeros(n)]
        if self._plan is not None:
            starts.insert(0, self._plan.copy())
        else:
            starts.append(np.concatenate([np.full(self.n_blocks, 0.5),
                                          np.full(self.n_blocks, 0.5)]))

        best, best_cost = None, np.inf
        for x0 in starts:
            res = minimize(obj, x0, jac=grad, method="L-BFGS-B",
                           bounds=[(0.0, 1.0)] * n,
                           options={"maxiter": self.max_iter, "ftol": 1e-6})
            if res.fun < best_cost:
                best, best_cost = res.x, float(res.fun)
        self.solves += 1
        return np.clip(best, 0.0, 1.0), best_cost

    # ------------------------------------------------------------------- act
    def act(self, row: dict, history: list):
        t_s = row["t_s"]
        if self._plan is None or t_s >= self._next_replan:
            lead = np.arange(self.n_steps) * PLAN_DT
            if self.use_forecast:
                t_out_series = self.forecast.at(t_s, lead)
            else:
                # Persistence: assume outdoor temperature holds where it is.
                t_out_series = np.full(self.n_steps, row["t_out"], dtype=float)
            # Solar and occupancy gains are predictable from the clock, so the
            # planner is allowed to know them. The weather is not, which is
            # why it arrives through a forecast with error.
            gain_series = np.array([self.gain_at(t_s + float(L)) for L in lead])
            plan, cost = self._solve(row["t_in"], row["t_wall"],
                                     t_out_series, gain_series, self.u)
            self._plan = plan
            self._next_replan = t_s + self.replan_s
            if self.record:
                self.trace.append({
                    "t_s": t_s, "cost": cost,
                    "u_plan": plan[:self.n_blocks].tolist(),
                    "f_plan": plan[self.n_blocks:].tolist(),
                    "forecast_peak": float(np.max(t_out_series)),
                    "forecast_min": float(np.min(t_out_series)),
                })

        u = float(self._plan[0])
        f = float(self._plan[self.n_blocks])
        # A barely-open window is not worth the actuation; snap it shut.
        if u < 0.03:
            u, f = 0.0, 0.0
        if f < 0.03:
            f = 0.0
        return self._set(t_s, u, f)
