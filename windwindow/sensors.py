"""
Step 5: virtual sensors.

Real temperature sensors lie in four different ways, and the controller has to
cope with all of them:

  offset        a fixed calibration error, different for each sensor and each
                time you rebuild the rig. Like a bathroom scale that always
                reads 0.3 kg heavy.
  lag           the sensor has its own thermal mass, so it shows you where the
                air was twenty seconds ago, not where it is now.
  noise         random jitter on every reading.
  quantisation  the reading arrives as a digital step, not a smooth number.
                0.0625 degC is the step of a 12-bit DS18B20.

The offset is drawn once per session and then held, which is what makes it
dangerous: it does not average away over a run, so a controller that compares
two sensors inherits the difference of their two offsets for the whole session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg

CHANNELS = ("t_in", "t_wall", "t_out")


@dataclass
class Sensor:
    """One noisy, laggy, offset thermometer."""

    offset: float
    tau_s: float = cfg.SENSOR_TAU_S
    noise_c: float = cfg.SENSOR_NOISE_C
    quant_c: float = cfg.SENSOR_QUANT_C
    state: float | None = None      # the lagged value the element has reached

    def update(self, true_value: float, dt: float) -> None:
        """Let the sensing element chase the true temperature for dt seconds."""
        if self.state is None:
            self.state = true_value
            return
        alpha = 1.0 - np.exp(-dt / self.tau_s)
        self.state += alpha * (true_value - self.state)

    def read(self, rng: np.random.Generator) -> float:
        """Take a reading: lagged value, plus offset, plus noise, then rounded."""
        v = self.state + self.offset + rng.normal(0.0, self.noise_c)
        return float(np.round(v / self.quant_c) * self.quant_c)


@dataclass
class SensorSuite:
    """The three thermometers in the rig, with per-session calibration errors."""

    seed: int = 0
    offset_sigma: float = cfg.SENSOR_OFFSET_C
    sensors: dict = field(default_factory=dict, repr=False)
    rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if not self.sensors:
            self.sensors = {
                c: Sensor(offset=float(self.rng.normal(0.0, self.offset_sigma)))
                for c in CHANNELS
            }

    @property
    def offsets(self) -> dict:
        return {c: s.offset for c, s in self.sensors.items()}

    def update(self, t_in: float, t_wall: float, t_out: float, dt: float) -> None:
        self.sensors["t_in"].update(t_in, dt)
        self.sensors["t_wall"].update(t_wall, dt)
        self.sensors["t_out"].update(t_out, dt)

    def read(self) -> dict:
        return {c: s.read(self.rng) for c, s in self.sensors.items()}
