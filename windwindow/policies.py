"""
Steps 7, 8 and 11: the controllers.

Four policies:

  TimerPolicy        ventilates on a fixed schedule, ignoring the weather.
                     This is the thing we are trying to beat on energy.
  SimpleRulePolicy   "open if it is cooler outside". The rule most people
                     would write, and the one PLAN goal G2 asks us to beat.
  RandomActionPolicy picks actions at random. Used only to collect training
                     data (Step 8) -- see the note below on why.
  ModelPolicy        WindWindow itself: predicts the next ten minutes for each
                     candidate action and picks the best. Added in Step 11.

A note on fairness. The timer and the baseline rule are given every advantage
that is not the model itself: the same minimum 5-minute gap between moves, the
same hysteresis band, and the same 24 degC floor. So the only thing that
differs between SimpleRulePolicy and ModelPolicy is *how they decide* -- one
compares two thermometers, the other predicts an outcome. Anything the model
wins by, it wins on the strength of its prediction.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from .features import state_from_history


class BasePolicy:
    """Shared plumbing: a name, the current action, and move rate-limiting."""

    name = "base"

    def __init__(self, min_dwell_s: float = cfg.MIN_DWELL_S):
        self.min_dwell_s = min_dwell_s
        self.u = 0.0
        self.f = 0.0
        self.last_move_t = -1e9

    def reset(self, scenario=None) -> None:
        self.u = 0.0
        self.f = 0.0
        self.last_move_t = -1e9

    def _may_move(self, t_s: float) -> bool:
        return (t_s - self.last_move_t) >= self.min_dwell_s

    def _set(self, t_s: float, u: float, f: float):
        """Change the window position, and remember when. Fan changes are free."""
        if abs(u - self.u) > 0.05:
            self.last_move_t = t_s
        self.u, self.f = u, f
        return u, f

    def act(self, row: dict, history: list):
        raise NotImplementedError


class TimerPolicy(BasePolicy):
    """Ventilate on a schedule, regardless of what the weather is doing.

    The naive automation: open the window and run the fan for the first half
    of every half-hour. It is the comparison point for goals G3 and G4 because
    it is what you get without any sensing at all.
    """

    name = "timer"

    def __init__(self, period_s: float = 30 * 60.0, duty: float = 0.5):
        super().__init__(min_dwell_s=0.0)      # the schedule is the rate limit
        self.period_s = period_s
        self.duty = duty

    def act(self, row: dict, history: list):
        phase = (row["t_s"] + cfg.DT_CONTROL) % self.period_s
        on = phase < self.duty * self.period_s
        return self._set(row["t_s"], 1.0 if on else 0.0, 1.0 if on else 0.0)


class SimpleRulePolicy(BasePolicy):
    """Open if it is cooler outside than inside.

    Two thresholds rather than one, so it does not flip-flop when the two
    readings are nearly equal: it opens once outside is OPEN_GAP cooler, and
    only closes again once that lead has shrunk to CLOSE_GAP.
    """

    name = "simple_rule"
    OPEN_GAP = 0.5      # degC cooler outside before opening
    CLOSE_GAP = 0.1     # degC; close once the lead falls below this

    def act(self, row: dict, history: list):
        t_s, t_in, t_out = row["t_s"], row["t_in"], row["t_out"]

        # Same comfort floor the model controller gets.
        if t_in < cfg.T_IN_FLOOR_C:
            return self._set(t_s, 0.0, 0.0) if self._may_move(t_s) else (self.u, self.f)

        gap = t_in - t_out          # how much cooler it is outside
        if not self._may_move(t_s):
            return self.u, self.f
        if self.u < 0.5 and gap > self.OPEN_GAP:
            return self._set(t_s, 1.0, 1.0)
        if self.u >= 0.5 and gap < self.CLOSE_GAP:
            return self._set(t_s, 0.0, 0.0)
        return self.u, self.f


class RandomActionPolicy(BasePolicy):
    """Hold a random action for a random stretch, then pick another.

    Why random, when random is obviously a bad way to run a room? Because of
    what the model has to learn. A sensible policy only ever opens the window
    when opening is a good idea, so a log of a sensible policy contains almost
    no examples of opening the window on a hot day. The model would then have
    no idea what happens if you do -- and predicting exactly that is its job.
    Random actions are how the training set gets to cover the bad ideas too.

    Holds are 6 to 20 minutes so that a decent share of rows have an action
    that stayed put across the whole 10-minute horizon (see
    features.featurise_session).
    """

    name = "random"

    def __init__(self, seed: int = 0, hold_min=(6.0, 20.0)):
        super().__init__(min_dwell_s=0.0)
        self.seed = seed
        self.hold_min = hold_min
        self.rng = np.random.default_rng(seed)
        self.until = -1.0

    def reset(self, scenario=None) -> None:
        super().reset(scenario)
        self.rng = np.random.default_rng(self.seed)
        self.until = -1.0
        self.u, self.f = self._draw()

    def _draw(self):
        u = float(self.rng.choice([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]))
        f = 0.0 if u < 0.05 else float(self.rng.choice([0.0, 0.5, 1.0]))
        return u, f

    def act(self, row: dict, history: list):
        t_s = row["t_s"]
        if t_s >= self.until:
            self.until = t_s + 60.0 * float(self.rng.uniform(*self.hold_min))
            u, f = self._draw()
            return self._set(t_s, u, f)
        return self.u, self.f


class ModelPolicy(BasePolicy):
    """Step 11: WindWindow. Predicts the next ten minutes, then picks.

    The decision rule is PLAN section 5, verbatim:

        benefit = predict(shut) - predict(open + fan)
        open   if benefit > 0.15 degC
        close  if benefit < 0.05 degC
        at least 5 min between moves
        close  if T_in < 24 degC
        fan on only if it beats window-only by more than 0.1 degC

    Two different thresholds for opening and closing (0.15 and 0.05) create a
    dead band. Without it, a benefit hovering around a single threshold would
    have the window opening and shutting every half minute.
    """

    name = "model"

    def __init__(self, predictor, min_dwell_s: float = cfg.MIN_DWELL_S,
                 record: bool = False, fan_margin: float = cfg.FAN_MARGIN_C,
                 name: str | None = None):
        super().__init__(min_dwell_s=min_dwell_s)
        self.predictor = predictor
        # How much extra cooling the fan must buy before it is worth switching
        # on. PLAN section 5 sets this to 0.10 degC as a first guess. It is the
        # single knob that trades comfort against fan energy, so Step 11 sweeps
        # it and Step 12 reports the chosen value on unseen weather.
        self.fan_margin = float(fan_margin)
        if name:
            self.name = name
        self.last_decision: dict = {}
        self.record = record
        self.trace: list = []      # one row per tick, for the Step 11 figure

    def reset(self, scenario=None) -> None:
        super().reset(scenario)
        self.trace = []
        self.last_decision = {}

    def act(self, row: dict, history: list):
        t_s = row["t_s"]
        state = state_from_history(row, history)

        # Predicted change in indoor temperature over the next 10 minutes,
        # for each action we could take.
        p_shut = self.predictor(state, 0.0, 0.0)
        p_open = self.predictor(state, 1.0, 0.0)
        p_open_fan = self.predictor(state, 1.0, 1.0)

        benefit = p_shut - min(p_open, p_open_fan)
        fan_gain = p_open - p_open_fan
        want_fan = 1.0 if fan_gain > self.fan_margin else 0.0

        self.last_decision = {
            "p_shut": p_shut, "p_open": p_open, "p_open_fan": p_open_fan,
            "benefit": benefit, "fan_gain": fan_gain,
        }
        if self.record:
            self.trace.append({"t_s": t_s, **self.last_decision,
                               "slope_in": state["slope_in"],
                               "slope_out": state["slope_out"]})

        # Comfort floor: never keep ventilating a room that is already cold.
        if row["t_in"] < cfg.T_IN_FLOOR_C:
            if self.u >= 0.5 and self._may_move(t_s):
                return self._set(t_s, 0.0, 0.0)
            if self.u < 0.5:
                return self._set(t_s, 0.0, 0.0)
            return self.u, self.f

        if self.u < 0.5:
            if benefit > cfg.OPEN_THRESHOLD_C and self._may_move(t_s):
                return self._set(t_s, 1.0, want_fan)
            return self.u, self.f

        # Already open: decide whether to close, and whether the fan still pays.
        if benefit < cfg.CLOSE_THRESHOLD_C and self._may_move(t_s):
            return self._set(t_s, 0.0, 0.0)
        self.f = want_fan          # the fan may change at any time; it is not a "move"
        return self.u, self.f


class FixedActionPolicy(BasePolicy):
    """Hold one action for the whole session. Used for Step 8 excitation runs.

    Random actions cover the action space but they do not cover the *state*
    space evenly, because the room keeps collapsing back toward equilibrium.
    In particular, "window open while it is hotter outside" puts itself out
    within about half an hour: the room simply warms up to the outdoor
    temperature and the case stops existing.

    So a handful of sessions instead start the room deliberately far from
    equilibrium and hold a single action for four hours. That is the standard
    way to identify a thermal system: pick a step input, then watch the whole
    response instead of a snippet of it. Every row of these sessions also has
    an action that held still across the horizon, so none of them get dropped
    by the action-stability filter.
    """

    name = "fixed"

    def __init__(self, u: float, f: float):
        super().__init__(min_dwell_s=0.0)
        self.fixed_u, self.fixed_f = float(u), float(f)

    def reset(self, scenario=None) -> None:
        super().reset(scenario)
        self.u, self.f = self.fixed_u, self.fixed_f

    def act(self, row: dict, history: list):
        return self.fixed_u, self.fixed_f
