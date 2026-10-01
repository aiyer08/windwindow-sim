"""
Step 3: the weather generator.

Produces an outdoor temperature trace, plus the state the room starts in.
Four named scenarios, matching PLAN section 4.

Each trace is a smooth base shape plus "gusts" -- a handful of sine waves at
different periods with random phases. Gusts matter for two reasons: they stop
the outdoor temperature from being a straight line the model could trivially
extrapolate, and they give slope_out (the 5-minute outdoor trend) something
real to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg

SCENARIOS = ("cool", "hot", "hot_then_cooling", "hot_walls_cold_outside")

# Gust periods in minutes, and the standard deviation of their sum in degC.
GUST_PERIODS_MIN = (7.0, 15.0, 31.0, 64.0)
GUST_SIGMA_C = 0.45


@dataclass
class Scenario:
    """One weather trace plus the room's starting condition."""

    name: str
    seed: int
    duration_s: float
    t_out_grid: np.ndarray = field(repr=False)   # sampled every DT_PHYSICS seconds
    t_in0: float
    t_wall0: float
    heater_w: float
    note: str = ""

    def t_out(self, t: float) -> float:
        """Outdoor temperature at time t seconds."""
        i = int(t / cfg.DT_PHYSICS)
        if i < 0:
            i = 0
        elif i >= len(self.t_out_grid):
            i = len(self.t_out_grid) - 1
        return float(self.t_out_grid[i])

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.t_out_grid)) * cfg.DT_PHYSICS


def _gusts(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A smooth, zero-mean wiggle with standard deviation GUST_SIGMA_C."""
    out = np.zeros_like(t)
    for period_min in GUST_PERIODS_MIN:
        phase = rng.uniform(0, 2 * np.pi)
        out += np.sin(2 * np.pi * t / (period_min * 60.0) + phase)
    # n independent sines of unit amplitude have std sqrt(n/2).
    return out * GUST_SIGMA_C / np.sqrt(len(GUST_PERIODS_MIN) / 2.0)


def make_scenario(
    name: str,
    seed: int = 0,
    duration_s: float = 3 * 3600.0,
    heater_w: float | None = None,
    t_in0: float | None = None,
    t_wall0: float | None = None,
) -> Scenario:
    """Build one scenario. Same (name, seed, duration) always gives the same trace.

    t_in0 and t_wall0 override where the room starts. Step 8 uses them to spread
    the training sessions across a much wider range of starting states than the
    four scenarios would give on their own -- in particular, starting the room
    well below the outdoor temperature, which is the only way to collect rows
    where the window is open while it is hotter outside.
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")

    # Deterministic across processes. Python's hash() of a string is salted
    # per process, so it must not be used here.
    rng = np.random.default_rng((SCENARIOS.index(name) + 1) * 100_003 + seed)
    t = np.arange(0.0, duration_s + cfg.DT_PHYSICS, cfg.DT_PHYSICS)
    frac = t / duration_s          # 0 at the start, 1 at the end
    gust = _gusts(t, rng)

    if name == "cool":
        # A mild day. Closed, the room drifts to about 24 degC -- under the
        # comfort threshold -- so the risk here is over-ventilating and
        # dropping below the 24 degC floor.
        base = 18.5 + 1.2 * np.sin(2 * np.pi * frac - 0.4) + rng.uniform(-0.8, 0.8)
        t_in0_default, t_wall0_default = 21.0 + rng.uniform(-0.6, 0.6), 20.0 + rng.uniform(-0.6, 0.6)
        note = "mild outdoor air; risk of cooling below the 24 °C floor"

    elif name == "hot":
        # Hot outside all day, starting from a room that is still cool from
        # overnight. Early on it is warmer outside than in, so opening would
        # warm the room and the right move is to stay shut. Later the heater
        # has pushed the room past the outdoor temperature and opening helps.
        base = 29.0 + 1.6 * np.sin(2 * np.pi * frac - 0.9) + rng.uniform(-0.8, 0.8)
        t_in0_default, t_wall0_default = 24.5 + rng.uniform(-0.8, 0.8), 24.0 + rng.uniform(-0.8, 0.8)
        note = "outdoor air warmer than indoor air until the crossover"

    elif name == "hot_then_cooling":
        # A hot afternoon giving way to a cool evening. The outdoor drop is
        # smooth but fast enough that the 5-minute outdoor trend carries real
        # information about what the next ten minutes hold.
        centre, width = 0.55, 0.13
        shape = 0.5 * (1.0 - np.tanh((frac - centre) / width))
        base = 19.0 + 11.5 * shape + rng.uniform(-0.8, 0.8)
        t_in0_default, t_wall0_default = 27.0 + rng.uniform(-0.8, 0.8), 28.5 + rng.uniform(-0.8, 0.8)
        note = "outdoor air falls about 11 K mid-run; the trend is informative"

    else:  # hot_walls_cold_outside
        # THE TRAP. The plywood shell has been baking all afternoon and starts
        # about 9 K hotter than the outdoor air, holding roughly 100 kJ of
        # stored heat. Opening the window flushes the *air* within a couple of
        # minutes, but the shell keeps re-warming it for the next half hour.
        # A controller that shuts the window as soon as the air reads "cool
        # enough" gets punished by the rebound. This is the scenario that needs
        # T_wall as a model input.
        #
        # The hot shell is set as a starting condition rather than by modelling
        # sunlight, which PLAN section 2 lists as a non-goal.
        base = 24.5 + 0.8 * np.sin(2 * np.pi * frac + 1.2) + rng.uniform(-0.6, 0.6)
        t_in0_default = 28.5 + rng.uniform(-0.7, 0.7)
        t_wall0_default = 34.0 + rng.uniform(-0.8, 0.8)
        note = "shell starts about 9 K above outdoor air; rebound on early closure"

    return Scenario(
        name=name,
        seed=seed,
        duration_s=duration_s,
        t_out_grid=base + gust,
        t_in0=float(t_in0) if t_in0 is not None else float(t_in0_default),
        t_wall0=float(t_wall0) if t_wall0 is not None else float(t_wall0_default),
        heater_w=cfg.HEATER_W if heater_w is None else float(heater_w),
        note=note,
    )
