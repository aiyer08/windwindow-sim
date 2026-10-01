"""
A two-state model of the room, fitted to logged data.

The first version of this project learned one number: how much the indoor
temperature would change over the next ten minutes. That's enough to compare
two actions once, which is all a greedy controller needs, but it can't answer
"where will this room be in four hours if I ventilate now", so it can't plan.

This fits the dynamics instead. The structure comes from physics, which is why
it's a grey-box model and not a black box: two thermal masses, heat moving
between them, and heat moving to outdoors through paths the window and fan open
up.

    dT_in/dt  = k_wall(u,f)*(T_wall - T_in) + k_vent(u,f)*(T_out - T_in)
                + gain/C_in
    dT_wall/dt = -k_wall(u,f)*(T_in - T_wall)*C_in/C_wall
                 + j_out*(T_out - T_wall)

Each conductance is allowed to vary with the action as k0 + k1*u + k2*u*f,
which is an approximation. In the simulated room the real dependence is
hyperbolic in the airflow, because the wall coupling is limited by a surface
film whose resistance drops as air moves over it. The approximation is within
about 3% across the action range, and `multistep_error` measures what that
costs over real horizons instead of assuming it's fine.

Ten parameters get fitted, all of them heat capacities or conductances, in log
space so they stay positive. The fit minimises error over multi-hour rollouts
rather than one-step differences. Both parts matter. One-step fitting is biased
here, because sensor noise in T_in sits on both sides of the equation, and it
optimises the wrong thing anyway: a planner needs trajectories that are right
hours out, not increments that are right over five minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg

PARAM_NAMES = [
    "C_in",       # J/K, air plus furnishings
    "C_wall",     # J/K, structure
    "K_iw_0",     # W/K, air to structure, window shut
    "K_iw_open",  # W/K, extra when the window is open
    "K_iw_fan",   # W/K, further extra with the fan running
    "K_leak",     # W/K, infiltration with the window shut
    "K_open",     # W/K, ventilation with the window fully open
    "K_fan",      # W/K, extra ventilation from the fan
    "K_wo",       # W/K, structure to outdoors
    "fan_heat",   # W, waste heat from the fan at full speed
]
N_PARAMS = len(PARAM_NAMES)

# Bounds, applied by fitting the logarithm of each parameter. Every quantity
# here is a heat capacity or a conductance, so all are strictly positive, and
# an earlier unconstrained version exploited that freedom: it returned a
# negative infiltration conductance and a structure 99% lighter than the real
# one, while still predicting the validation sessions acceptably. Parameters
# that cannot be given a physical reading cannot be trusted to extrapolate to
# action sequences the data did not contain, which is exactly what a planner
# asks of them.
LOWER = np.array([1e4, 1e5, 0.5, 0.5, 0.5, 0.2, 5.0, 5.0, 1.0, 1.0])
UPPER = np.array([2e6, 5e7, 500.0, 500.0, 500.0, 200.0, 2000.0, 2000.0, 500.0, 200.0])


def true_params() -> np.ndarray:
    """The parameters implied by config.py, for comparison only.

    Never used in fitting or control. Reported beside the identified values so
    the write-up can show how close identification came unaided.
    """
    from .room import k_in_wall

    return np.array([
        cfg.C_IN,
        cfg.C_WALL,
        cfg.K_IN_WALL_BASE,
        k_in_wall(1.0, 0.0) - cfg.K_IN_WALL_BASE,
        k_in_wall(1.0, 1.0) - k_in_wall(1.0, 0.0),
        cfg.K_LEAK,
        cfg.K_VENT_OPEN_MAX,
        cfg.K_VENT_FAN_MAX,
        cfg.K_WALL_OUT,
        cfg.FAN_HEAT_FRAC * cfg.FAN_ELEC_W,
    ])


def _rates(theta, t_in, t_wall, t_out, u, f, gain):
    """dT_in/dt and dT_wall/dt. Works on scalars or arrays.

    The same wall conductance appears in both equations, divided by each
    mass's own capacity. Enforcing that shared structure is what makes the
    parameters identifiable: their ratio is fixed by the capacities rather
    than free to absorb error.
    """
    (c_in, c_wall, k_iw0, k_iw_o, k_iw_f,
     k_leak, k_open, k_fan, k_wo, fan_heat) = theta
    uf = u * f
    k_iw = k_iw0 + k_iw_o * u + k_iw_f * uf
    k_vent = k_leak + k_open * u + k_fan * uf
    q_wall = k_iw * (t_wall - t_in)
    d_in = (q_wall + k_vent * (t_out - t_in) + gain + fan_heat * f) / c_in
    d_wall = (-q_wall + k_wo * (t_out - t_wall)) / c_wall
    return d_in, d_wall


@dataclass
class StateSpaceModel:
    """Identified room dynamics. Roll forward with `step` or `rollout`."""

    theta: np.ndarray = field(default_factory=true_params)
    fit_info: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- rollout
    def step(self, t_in, t_wall, t_out, u, f, gain, dt):
        """One second-order (midpoint) step. Vectorised over arrays."""
        a1, b1 = _rates(self.theta, t_in, t_wall, t_out, u, f, gain)
        a2, b2 = _rates(self.theta, t_in + 0.5 * dt * a1, t_wall + 0.5 * dt * b1,
                        t_out, u, f, gain)
        return t_in + dt * a2, t_wall + dt * b2

    def rollout(self, t_in, t_wall, t_out_series, u_series, f_series, heater,
                dt=cfg.DT_CONTROL):
        """heater may be a scalar or a series, one value per step."""
        """Simulate forward. Returns (T_in trace, T_wall trace), length n+1.

        t_in and t_wall may be arrays, in which case many trajectories are
        advanced at once and the series arguments are indexed along axis 0.
        """
        n = len(t_out_series)
        ti = np.asarray(t_in, dtype=float)
        tw = np.asarray(t_wall, dtype=float)
        out_i = [ti.copy()]
        out_w = [tw.copy()]
        heat_series = np.ndim(heater) > 0 and len(np.atleast_1d(heater)) == n
        for k in range(n):
            q = heater[k] if heat_series else heater
            ti, tw = self.step(ti, tw, t_out_series[k], u_series[k], f_series[k],
                               q, dt)
            out_i.append(ti.copy() if np.ndim(ti) else ti)
            out_w.append(tw.copy() if np.ndim(tw) else tw)
        return np.array(out_i), np.array(out_w)

    # ------------------------------------------------------------------ fit
    def fit(self, sessions, horizons_min=(60.0, 360.0), stride_min=45.0,
            verbose=True, x0=None):
        """Identify the parameters from a list of logged session frames.

        Each frame needs the measured columns t_in, t_wall, t_out plus u, f and
        heater_w, sampled at the control interval.

        Fitting minimises error over multi-hour rollouts rather than one-step
        differences. One-step fitting is biased here, because sensor noise in
        T_in sits on both sides of the equation, and it optimises the wrong
        quantity: a planner needs trajectories that are right hours out, not
        increments that are right over five minutes.
        """
        from scipy.optimize import least_squares

        # Windows at more than one length, scored together. Fitting only short
        # windows leaves the slow structural mode barely excited; fitting only
        # long ones lets short-term error grow unchecked. Including both in one
        # residual vector avoids having to choose, and avoids choosing by
        # looking at the held-out sessions, which would spend them.
        stride = max(1, int(round(stride_min * 60.0 / cfg.DT_CONTROL)))
        groups = []
        for horizon_min in horizons_min:
            H = int(round(horizon_min * 60.0 / cfg.DT_CONTROL))
            groups.append((H, self._windows(sessions, H, stride)))

        if x0 is None:
            x0 = np.array([1.0e5, 1.0e6, 20.0, 20.0, 20.0,
                           10.0, 100.0, 100.0, 20.0, 20.0])

        def residuals(log_theta):
            self.theta = np.exp(log_theta)
            out = []
            for H, w in groups:
                si, sw, seq_out, seq_u, seq_f, seq_g, ti_t, tw_t = w
                ti = si.copy(); tw = sw.copy()
                checks = np.arange(2, H, max(1, H // 12))
                c = 0
                for k in range(H):
                    ti, tw = self.step(ti, tw, seq_out[k], seq_u[k], seq_f[k],
                                       seq_g[k], cfg.DT_CONTROL)
                    if c < len(checks) and k == checks[c]:
                        out.append(ti - ti_t[k])
                        out.append(tw - tw_t[k])
                        c += 1
            return np.concatenate(out)

        n_win = sum(len(w[0]) for _, w in groups)
        if verbose:
            print(f"    fitting 10 physical parameters on {n_win:,} rollouts "
                  f"at horizons {tuple(int(h) for h in horizons_min)} min")
        sol = least_squares(
            residuals, np.log(x0), method="trf", x_scale="jac",
            bounds=(np.log(LOWER), np.log(UPPER)),
            xtol=1e-12, ftol=1e-12, max_nfev=400)
        self.theta = np.exp(sol.x)
        r = residuals(sol.x)
        self.fit_info = {
            "n_sessions": len(sessions), "n_windows": int(n_win),
            "horizons_min": list(horizons_min),
            "final_rollout_rmse": float(np.sqrt(np.mean(r ** 2))),
            "success": bool(sol.success), "n_fev": int(sol.nfev),
            "x0": np.asarray(x0).tolist(),
        }
        return self

    @staticmethod
    def _windows(sessions, H, stride):
        """Slice sessions into rollout windows of H steps."""
        si, sw, seq_out, seq_u, seq_f, seq_g, ti_t, tw_t = [], [], [], [], [], [], [], []
        for df in sessions:
            ti = df["t_in"].to_numpy(float)
            tw = df["t_wall"].to_numpy(float)
            to = df["t_out"].to_numpy(float)
            u = df["u"].to_numpy(float)
            f = df["f"].to_numpy(float)
            g = df["heater_w"].to_numpy(float)
            for a in range(0, len(ti) - H - 1, stride):
                si.append(ti[a]); sw.append(tw[a])
                seq_out.append(to[a:a + H]); seq_u.append(u[a:a + H])
                seq_f.append(f[a:a + H]); seq_g.append(g[a:a + H])
                ti_t.append(ti[a + 1:a + H + 1]); tw_t.append(tw[a + 1:a + H + 1])
        return (np.array(si), np.array(sw), np.array(seq_out).T,
                np.array(seq_u).T, np.array(seq_f).T, np.array(seq_g).T,
                np.array(ti_t).T, np.array(tw_t).T)

    # -------------------------------------------------------------- validate
    def multistep_error(self, sessions, horizons_min=(10, 60, 180, 360)):
        """Prediction error against measured data, by horizon.

        Rolls the model from each start point using the true action and
        outdoor series, then compares with what the room actually did.
        """
        rows = []
        for hm in horizons_min:
            H = int(round(hm * 60.0 / cfg.DT_CONTROL))
            errs = []
            for df in sessions:
                ti = df["t_in"].to_numpy(float)
                tw = df["t_wall"].to_numpy(float)
                to = df["t_out"].to_numpy(float)
                u = df["u"].to_numpy(float)
                f = df["f"].to_numpy(float)
                q = float(df["heater_w"].iloc[0])
                stride = max(1, H // 4)
                for s in range(0, len(ti) - H - 1, stride):
                    pi, _ = self.rollout(ti[s], tw[s], to[s:s + H],
                                         u[s:s + H], f[s:s + H], q)
                    errs.append(pi[-1] - ti[s + H])
            e = np.array(errs)
            rows.append({"horizon_min": hm, "n": len(e),
                         "mae": float(np.mean(np.abs(e))),
                         "rmse": float(np.sqrt(np.mean(e ** 2))),
                         "bias": float(np.mean(e))})
        return rows

    def as_dict(self) -> dict:
        return {n: float(v) for n, v in zip(PARAM_NAMES, self.theta)}
