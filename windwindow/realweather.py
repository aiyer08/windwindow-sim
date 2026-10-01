"""
Real hourly weather, cached to CSV.

The first version of this used synthetic three-hour temperature slices. Fine
for checking the physics, but they never contained the thing that matters most
for ventilation control: a cool night followed by a hot afternoon. With that
pattern present, a controller that knows what's coming can flush heat out of
the structure overnight and then shut the room up during the peak. Without it
there's nothing to plan for.

Data comes from the Open-Meteo historical archive, which needs no API key.
scripts/fetch_weather.py downloads it once into data/weather, so everything
afterwards runs offline and every result reproduces from files in the repo.

Hourly readings get interpolated to the simulation step. Interpolation on its
own gives an unrealistically smooth series, so a small turbulent component goes
on top, with roughly the amplitude and timescale you measure at a weather
station on a sunny afternoon.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

WEATHER_DIR = pathlib.Path("data/weather")

# Five climates chosen to span the range of situations a ventilation
# controller meets, rather than five variations on one. The dates are real
# heat events; "night_drop" is the mean overnight fall, which is what decides
# whether pre-cooling is available at all.
LOCATIONS = {
    "palo_alto": dict(
        lat=37.43, lon=-122.17, tz="America/Los_Angeles",
        start="2024-07-08", end="2024-07-10",
        label="Palo Alto, CA", note="Mediterranean; large diurnal swing"),
    "phoenix": dict(
        lat=33.45, lon=-112.07, tz="America/Phoenix",
        start="2024-07-09", end="2024-07-11",
        label="Phoenix, AZ", note="Desert heat wave; hot nights"),
    "austin": dict(
        lat=30.27, lon=-97.74, tz="America/Chicago",
        start="2024-08-04", end="2024-08-06",
        label="Austin, TX", note="Hot and humid; warm nights"),
    "chicago": dict(
        lat=41.88, lon=-87.63, tz="America/Chicago",
        start="2024-06-17", end="2024-06-19",
        label="Chicago, IL", note="Continental summer heat spike"),
    "seattle": dict(
        lat=47.61, lon=-122.33, tz="America/Los_Angeles",
        start="2024-07-05", end="2024-07-07",
        label="Seattle, WA", note="Cool maritime; ventilation almost always helps"),
}

# Sub-hourly turbulence. Periods in minutes; sigma in degC. Outdoor air at a
# fixed point fluctuates by a few tenths of a degree over minutes on a sunny
# day, much less at night, which the daytime scaling below reproduces.
TURB_PERIODS_MIN = (9.0, 23.0, 47.0)
TURB_SIGMA_C = 0.25


def archive_url(loc: dict) -> str:
    return (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&start_date={loc['start']}&end_date={loc['end']}"
        "&hourly=temperature_2m,relative_humidity_2m,cloud_cover"
        f"&timezone={loc['tz'].replace('/', '%2F')}"
    )


def load_hourly(name: str) -> pd.DataFrame:
    """Read one cached location. Columns: time, temperature_2m, hour_of_day."""
    path = WEATHER_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/fetch_weather.py once to download it.")
    df = pd.read_csv(path, parse_dates=["time"])
    df["hour_of_day"] = df["time"].dt.hour + df["time"].dt.minute / 60.0
    return df


@dataclass
class RealWeather:
    """An outdoor temperature series at simulation resolution."""

    name: str
    label: str
    note: str
    t_out_grid: np.ndarray = field(repr=False)
    start_time: pd.Timestamp = None
    duration_s: float = 0.0
    seed: int = 0
    # The simulator takes the starting room state and the heat load from the
    # weather object, so these carry the same names Scenario uses and
    # simulator.run_session works with either without changes.
    t_in0: float = 24.0
    t_wall0: float = 24.0
    heater_w: float = cfg.HEATER_W

    def t_out(self, t: float) -> float:
        i = int(t / cfg.DT_PHYSICS)
        if i < 0:
            i = 0
        elif i >= len(self.t_out_grid):
            i = len(self.t_out_grid) - 1
        return float(self.t_out_grid[i])

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.t_out_grid)) * cfg.DT_PHYSICS

    def gain_w(self, t: float) -> float:
        """Heat gain at time t, from the clock time it corresponds to."""
        from .gains import total_w

        h0 = self.start_time.hour + self.start_time.minute / 60.0
        return float(total_w(h0 + t / 3600.0))

    def clock_hours(self) -> np.ndarray:
        """Hour of day for each sample, for plotting against a real clock."""
        h0 = self.start_time.hour + self.start_time.minute / 60.0
        return h0 + self.times / 3600.0

    def summary(self) -> dict:
        g = self.t_out_grid
        hours = self.clock_hours() % 24
        night = (hours < 6) | (hours >= 22)
        return {
            "location": self.name,
            "label": self.label,
            "hours": len(g) * cfg.DT_PHYSICS / 3600.0,
            "t_out_min": float(g.min()),
            "t_out_max": float(g.max()),
            "t_out_mean": float(g.mean()),
            "diurnal_swing": float(g.max() - g.min()),
            "night_mean": float(g[night].mean()) if night.any() else float("nan"),
            "hours_above_26": float((g > cfg.COMFORT_C).sum() * cfg.DT_PHYSICS / 3600.0),
        }


def build(name: str, seed: int = 0, hours: float | None = None,
           start_hour: int = 0, turbulence: bool = True,
           t_in0: float | None = None, t_wall0: float | None = None,
           heater_w: float | None = None) -> RealWeather:
    """Interpolate a cached location to simulation resolution.

    start_hour lets a run begin at a chosen time of day, which matters because
    a controller that starts at midnight has the whole cool night available to
    pre-cool with, and one that starts at noon does not.
    """
    meta = LOCATIONS[name]
    df = load_hourly(name)

    # Trim to the requested window.
    t0 = df["time"].iloc[0].normalize() + pd.Timedelta(hours=start_hour)
    df = df[df["time"] >= t0].reset_index(drop=True)
    if hours is not None:
        df = df[df["time"] <= t0 + pd.Timedelta(hours=hours)].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(f"not enough cached hours for {name} from hour {start_hour}")

    duration_s = (len(df) - 1) * 3600.0
    n = int(duration_s / cfg.DT_PHYSICS) + 1
    t = np.arange(n) * cfg.DT_PHYSICS

    # Hourly readings are point samples, so a smooth interpolation through them
    # is the honest reconstruction.
    hours_axis = np.arange(len(df)) * 3600.0
    base = np.interp(t, hours_axis, df["temperature_2m"].to_numpy(dtype=float))

    if turbulence:
        rng = np.random.default_rng(abs(hash(name)) % 1000 + seed * 7919)
        turb = np.zeros_like(t)
        for period_min in TURB_PERIODS_MIN:
            turb += np.sin(2 * np.pi * t / (period_min * 60.0) + rng.uniform(0, 2 * np.pi))
        turb *= TURB_SIGMA_C / np.sqrt(len(TURB_PERIODS_MIN) / 2.0)
        # Turbulence is driven by surface heating, so scale it down at night.
        clock = (t0.hour + t0.minute / 60.0 + t / 3600.0) % 24
        daytime = 0.25 + 0.75 * np.clip(np.sin(np.pi * (clock - 6) / 12.0), 0, 1)
        base = base + turb * daytime

    # Default the room to the outdoor temperature at the start, which is what a
    # room left alone overnight would actually be near.
    start_out = float(base[0])
    return RealWeather(
        name=name, label=meta["label"], note=meta["note"],
        t_out_grid=base, start_time=t0, duration_s=duration_s, seed=seed,
        t_in0=start_out + 2.0 if t_in0 is None else float(t_in0),
        t_wall0=start_out + 1.0 if t_wall0 is None else float(t_wall0),
        heater_w=cfg.HEATER_W if heater_w is None else float(heater_w),
    )
