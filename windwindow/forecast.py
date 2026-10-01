"""
A weather forecast with realistic error.

A controller planning hours ahead needs to know what the outdoor temperature is
going to do. In a real install that comes from a free public forecast API,
which is wrong by an amount that grows the further out you look. Operational
two-metre temperature forecasts are typically off by about 1 degC at six hours
and 2 degC at two days.

Handing the controller the true future would be cheating and would overstate
what a real install achieves, so the error is modelled:

    forecast(now, now + h) = truth(now + h) + SIGMA_MAX * e(now + h) * ramp(h)

`e` is a smooth process over absolute time with unit variance, so the error is
correlated from one lead time to the next rather than independent, the way a
real forecast runs warm or cool for a stretch. `ramp` rises from zero, because
current conditions are measured rather than forecast, to one at long lead
times. One consequence worth noting: the forecast for a given hour improves as
that hour gets closer, which is also how real forecasts behave, and it's what
makes re-planning worth doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg

SIGMA_MAX = 2.0          # degC, error standard deviation at long lead times
RAMP_HOURS = 18.0        # lead time at which the error reaches SIGMA_MAX
ERROR_PERIODS_H = (7.0, 19.0, 41.0)   # timescales of the error process


def ramp(lead_s: np.ndarray | float) -> np.ndarray | float:
    """Fraction of the full error applying at a given lead time."""
    h = np.asarray(lead_s, dtype=float) / 3600.0
    return np.clip(np.sqrt(h / RAMP_HOURS), 0.0, 1.0)


@dataclass
class Forecast:
    """Wraps a true outdoor series and serves an imperfect view of it."""

    truth: np.ndarray = field(repr=False)
    error_field: np.ndarray = field(repr=False)
    seed: int = 0
    perfect: bool = False

    @classmethod
    def build(cls, truth: np.ndarray, seed: int = 0, perfect: bool = False):
        t = np.arange(len(truth)) * cfg.DT_PHYSICS
        rng = np.random.default_rng(700_000 + seed)
        e = np.zeros_like(t, dtype=float)
        for period_h in ERROR_PERIODS_H:
            e += np.sin(2 * np.pi * t / (period_h * 3600.0) + rng.uniform(0, 2 * np.pi))
        e *= 1.0 / np.sqrt(len(ERROR_PERIODS_H) / 2.0)      # unit variance
        e += rng.normal(0.0, 0.35)                           # a standing bias
        return cls(truth=truth, error_field=e, seed=seed, perfect=perfect)

    def at(self, now_s: float, lead_s: np.ndarray) -> np.ndarray:
        """The forecast issued at `now_s` for the given lead times."""
        idx = np.clip(((now_s + lead_s) / cfg.DT_PHYSICS).astype(int),
                      0, len(self.truth) - 1)
        true = self.truth[idx]
        if self.perfect:
            return true
        return true + SIGMA_MAX * self.error_field[idx] * ramp(lead_s)

    def error_profile(self, leads_h=(1, 3, 6, 12, 24)) -> list:
        """Measured error of this forecast, by lead time, for reporting."""
        rows = []
        n = len(self.truth)
        for lh in leads_h:
            lead = lh * 3600.0
            step = int(3600.0 / cfg.DT_PHYSICS)
            nows = np.arange(0, max(1, n - int(lead / cfg.DT_PHYSICS)), step,
                             dtype=float) * cfg.DT_PHYSICS
            if len(nows) == 0:
                continue
            pred = np.array([self.at(t, np.array([lead]))[0] for t in nows])
            idx = np.clip(((nows + lead) / cfg.DT_PHYSICS).astype(int), 0, n - 1)
            err = pred - self.truth[idx]
            rows.append({"lead_hours": lh, "n": len(err),
                         "mae": float(np.mean(np.abs(err))),
                         "rmse": float(np.sqrt(np.mean(err ** 2)))})
        return rows
