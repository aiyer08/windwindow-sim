"""
Heat gains: people, equipment and sun.

The old box had one constant 17.6 W heater. A real room is driven by gains that
move through the day, and the shape of that movement is most of what makes
ventilation timing matter: the sun peaks in the late afternoon, occupancy peaks
in the evening, and both are near zero overnight, which is exactly when the
outside air is coldest.

These are plain functions of clock time. The controller is allowed to know
them, which is fair enough: solar geometry follows from the date, and an
occupancy schedule is either known or learnable. Treating them as unknown would
undersell what a real install can do, and treating the weather as known would
oversell it, which is why the weather gets a forecast with error and the gains
don't.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg


def internal_w(hour_of_day):
    """Occupancy and equipment, W. Base load plus an evening peak."""
    h = np.asarray(hour_of_day, dtype=float) % 24.0
    evening = cfg.GAIN_EVENING_W * np.exp(
        -(((h - cfg.GAIN_EVENING_HOUR) / cfg.GAIN_EVENING_WIDTH_H) ** 2))
    return cfg.GAIN_BASE_W + evening


def solar_w(hour_of_day):
    """Solar gain through the window, W. Zero at night, peaking mid-afternoon."""
    h = np.asarray(hour_of_day, dtype=float) % 24.0
    daylight = (h >= cfg.SOLAR_FIRST_HOUR) & (h <= cfg.SOLAR_LAST_HOUR)
    shape = np.exp(-(((h - cfg.SOLAR_PEAK_HOUR) / cfg.SOLAR_WIDTH_H) ** 2))
    return np.where(daylight, cfg.SOLAR_PEAK_W * shape, 0.0)


def total_w(hour_of_day):
    """Internal plus solar gain, W."""
    return internal_w(hour_of_day) + solar_w(hour_of_day)


def daily_summary() -> dict:
    h = np.arange(0, 24, 1 / 60)
    g = total_w(h)
    return {
        "mean_w": float(g.mean()),
        "peak_w": float(g.max()),
        "peak_hour": float(h[int(np.argmax(g))]),
        "min_w": float(g.min()),
        "kwh_per_day": float(g.mean() * 24 / 1000.0),
    }
