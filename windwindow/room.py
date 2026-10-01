"""
Step 1-2: the virtual room.

Two temperatures change over time:

  T_in    the air inside the box
  T_wall  the plywood shell, which stores a lot of heat

The outdoor temperature T_out is given to us by the weather generator; it is
not affected by the room.

The equations are just "heat in minus heat out, divided by how much heat it
takes to warm the thing by one degree":

  C_in  * dT_in/dt   = Q_heater + Q_fan
                       + K_in_wall(u,f) * (T_wall - T_in)
                       + (K_vent(u,f) + K_leak) * (T_out - T_in)

  C_wall * dT_wall/dt = K_in_wall(u,f) * (T_in - T_wall)
                        + K_wall_out * (T_out - T_wall)

u is how open the window is (0 = shut, 1 = wide open).
f is the fan speed (0 = off, 1 = full).

For a fixed u, f and T_out this is a *linear* system, which is what lets
Step 2 check the simulator against an exact hand calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg


def mix_fraction(u: float, f: float) -> float:
    """How much of full forced airflow is moving over the inside surfaces, 0-1.

    The window has to be open for air to move at all, hence the factor of u.
    An open window on its own gets you part of the way; the fan does the rest.
    """
    m = u * (cfg.MIX_WINDOW + (1.0 - cfg.MIX_WINDOW) * f)
    return 0.0 if m < 0.0 else (1.0 if m > 1.0 else m)


def h_in_eff(u: float, f: float) -> float:
    """Inside surface film coefficient, W/(m^2.K). Rises as air starts moving."""
    return cfg.H_IN + (cfg.H_IN_FORCED - cfg.H_IN) * mix_fraction(u, f)


def k_in_wall(u: float, f: float) -> float:
    """Air <-> wall conductance, W/K.

    The inside air film in series with half the plywood. Opening the window
    stirs the air, which shrinks the film resistance and speeds up heat
    transfer between the walls and the air. This is the effect that lets hot
    walls re-warm the room when the window is open.
    """
    return cfg.AREA_WALL / (1.0 / h_in_eff(u, f) + cfg.R_HALF_WALL)


def k_vent(u: float, f: float) -> float:
    """Air <-> outdoors conductance through the window opening, W/K.

    The fan can only move air if the window is open, hence the factor of u
    on both terms.
    """
    return u * (cfg.K_VENT_OPEN_MAX + f * cfg.K_VENT_FAN_MAX)


def fan_heat_w(f: float) -> float:
    """Waste heat the fan motor dumps into the room, W."""
    return cfg.FAN_HEAT_FRAC * cfg.FAN_ELEC_W * f


def fan_elec_w(f: float) -> float:
    """Electrical power the fan draws, W."""
    return cfg.FAN_ELEC_W * f


def linear_system(u: float, f: float, t_out: float, heater_w: float):
    """Return (A, b) so that d[T_in, T_wall]/dt = A @ [T_in, T_wall] + b.

    Exposed so Step 2 can compare the simulator against an exact matrix
    exponential solution.
    """
    kiw = k_in_wall(u, f)
    kv = k_vent(u, f) + cfg.K_LEAK
    a = np.array(
        [
            [-(kiw + kv) / cfg.C_IN, kiw / cfg.C_IN],
            [kiw / cfg.C_WALL, -(kiw + cfg.K_WALL_OUT) / cfg.C_WALL],
        ]
    )
    b = np.array(
        [
            (heater_w + fan_heat_w(f) + kv * t_out) / cfg.C_IN,
            (cfg.K_WALL_OUT * t_out) / cfg.C_WALL,
        ]
    )
    return a, b


def steady_state(u: float, f: float, t_out: float, heater_w: float):
    """The temperatures the room would eventually settle at. Solves A x + b = 0."""
    a, b = linear_system(u, f, t_out, heater_w)
    return np.linalg.solve(a, -b)


@dataclass
class Room:
    """The virtual room. Call step() to move time forward."""

    t_in: float = 22.0
    t_wall: float = 22.0
    heater_w: float = cfg.HEATER_W

    # Running totals, for the energy goal G4.
    fan_wh: float = 0.0
    fan_seconds: float = 0.0
    heater_wh: float = 0.0
    elapsed_s: float = 0.0

    history: list = field(default_factory=list, repr=False)

    def derivatives(self, t_in: float, t_wall: float, u: float, f: float,
                    t_out: float, gain_w: float | None = None):
        """The two dT/dt values, in K/s.

        gain_w is the current heat gain from occupants, equipment and sun. It
        varies through the day in a real room, so it is passed in rather than
        stored. Omitting it falls back to the constant load in self.heater_w,
        which is what the legacy box scripts use.
        """
        q_gain = self.heater_w if gain_w is None else gain_w
        kiw = k_in_wall(u, f)
        kv = k_vent(u, f) + cfg.K_LEAK
        q_in = (
            q_gain
            + fan_heat_w(f)
            + kiw * (t_wall - t_in)
            + kv * (t_out - t_in)
        )
        q_wall = kiw * (t_in - t_wall) + cfg.K_WALL_OUT * (t_out - t_wall)
        return q_in / cfg.C_IN, q_wall / cfg.C_WALL

    def step(self, u: float, f: float, t_out: float, dt: float = cfg.DT_PHYSICS,
             gain_w: float | None = None) -> None:
        """Advance the room by dt seconds using 4th-order Runge-Kutta.

        Written out in plain floats rather than numpy arrays: the state is only
        two numbers, so array overhead would dominate, and a full experiment
        runs a few million of these steps.
        """
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0

        kiw = k_in_wall(u, f)
        kv = u * (cfg.K_VENT_OPEN_MAX + f * cfg.K_VENT_FAN_MAX) + cfg.K_LEAK
        kwo = cfg.K_WALL_OUT
        c_in, c_wall = cfg.C_IN, cfg.C_WALL
        q_gain = self.heater_w if gain_w is None else gain_w
        q_src = q_gain + cfg.FAN_HEAT_FRAC * cfg.FAN_ELEC_W * f

        def d(ti, tw):
            return ((q_src + kiw * (tw - ti) + kv * (t_out - ti)) / c_in,
                    (kiw * (ti - tw) + kwo * (t_out - tw)) / c_wall)

        ti, tw = self.t_in, self.t_wall
        a1, b1 = d(ti, tw)
        a2, b2 = d(ti + 0.5 * dt * a1, tw + 0.5 * dt * b1)
        a3, b3 = d(ti + 0.5 * dt * a2, tw + 0.5 * dt * b2)
        a4, b4 = d(ti + dt * a3, tw + dt * b3)
        self.t_in = ti + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        self.t_wall = tw + (dt / 6.0) * (b1 + 2.0 * b2 + 2.0 * b3 + b4)

        self.fan_wh += cfg.FAN_ELEC_W * f * dt / 3600.0
        self.heater_wh += q_gain * dt / 3600.0
        if f > 0.05:
            self.fan_seconds += dt
        self.elapsed_s += dt

    def copy(self) -> "Room":
        """A detached clone. Used for the paired what-if trials in Step 13."""
        return Room(
            t_in=self.t_in,
            t_wall=self.t_wall,
            heater_w=self.heater_w,
            fan_wh=self.fan_wh,
            fan_seconds=self.fan_seconds,
            heater_wh=self.heater_wh,
            elapsed_s=self.elapsed_s,
        )


# --------------------------------------------------------------------------
# Exact rollout
# --------------------------------------------------------------------------
_EXPM_CACHE: dict = {}


def rollout_exact(t_in, t_wall, u, f, t_out, heater_w, seconds):
    """Where the room lands after `seconds`, computed exactly.

    While the action (u, f) and the outdoor temperature are held fixed, the
    room is a linear system, so it has a closed-form answer and we do not need
    to step through it. Used as the physics "oracle" for the paired what-if
    trials in Step 13, and to probe the decision boundary.

    Returns (T_in, T_wall) at the end.
    """
    from scipy.linalg import expm

    key = (round(u, 6), round(f, 6), round(seconds, 6))
    m = _EXPM_CACHE.get(key)
    if m is None:
        a, _ = linear_system(u, f, 0.0, 0.0)      # A does not depend on t_out or heater
        m = expm(a * seconds)
        _EXPM_CACHE[key] = m

    a, b = linear_system(u, f, t_out, heater_w)
    x_ss = np.linalg.solve(a, -b)
    x = m @ (np.array([t_in, t_wall]) - x_ss) + x_ss
    return float(x[0]), float(x[1])
