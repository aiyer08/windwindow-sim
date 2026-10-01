"""
Step 9: model inputs, labels, and the train/test split.

The nine inputs are exactly the ones listed in PLAN section 5. Two properties
of this list matter:

1. Every input is built from (state, action). So once the model is trained we
   can ask it "what would happen if the window were shut?" and "what would
   happen if the window were open with the fan on?" from the *same* state.
   That counterfactual is the whole basis of the controller in Step 11.

2. Four of the nine carry the action u as a factor, and they all go to zero
   when the window is shut. That is deliberate: those terms describe things
   that only happen when air is moving -- outside air coming in, and the walls
   giving up their stored heat faster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

FEATURE_NAMES = [
    "d",                # outdoor minus indoor: which way heat wants to move
    "u_d",              # u * d: that flow, but only if the window is open
    "u_f_d",            # u * f * d: and boosted by the fan
    "wall_minus_in",    # T_wall - T_in: heat stored in the shell
    "heater_w",         # the internal heat load
    "slope_in",         # K/min, indoor trend over the last 5 min
    "slope_out",        # K/min, outdoor trend over the last 5 min
    "u_wall_minus_out", # u * (T_wall - T_out): walls re-warming the incoming air
    "u_slope_out",      # u * slope_out: where the outdoor air is heading
]

# Human-readable names, for the figures.
FEATURE_LABELS = {
    "d": "T_out - T_in",
    "u_d": "u . (T_out - T_in)",
    "u_f_d": "u . f . (T_out - T_in)",
    "wall_minus_in": "T_wall - T_in",
    "heater_w": "heater (W)",
    "slope_in": "indoor trend",
    "slope_out": "outdoor trend",
    "u_wall_minus_out": "u . (T_wall - T_out)",
    "u_slope_out": "u . outdoor trend",
}


def build_features(t_in, t_wall, t_out, heater_w, slope_in, slope_out, u, f):
    """The nine inputs for one (state, action) pair, in FEATURE_NAMES order.

    Works on scalars or on numpy arrays, so it serves both the controller
    (one state at a time) and training (a whole column at once).
    """
    d = t_out - t_in
    return [
        d,
        u * d,
        u * f * d,
        t_wall - t_in,
        heater_w,
        slope_in,
        slope_out,
        u * (t_wall - t_out),
        u * slope_out,
    ]


def feature_vector(state: dict, u: float, f: float) -> np.ndarray:
    """One row, shaped (1, 9), ready to hand to a fitted model."""
    return np.array([build_features(
        state["t_in"], state["t_wall"], state["t_out"], state["heater_w"],
        state["slope_in"], state["slope_out"], u, f,
    )], dtype=float)


def slope_per_min(history: list, key: str, window_s: float = cfg.SLOPE_WINDOW_S) -> float:
    """Trend of `key` over the last `window_s` seconds, in degC per minute.

    Fitted by least squares across the whole window rather than taking the
    difference of the two end points, because the end points carry sensor
    noise and a fit averages it down.
    """
    if not history:
        return 0.0
    t_end = history[-1]["t_s"]
    pts = [(r["t_s"], r[key]) for r in history if t_end - r["t_s"] <= window_s]
    if len(pts) < 3:
        return 0.0
    t = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    if t.max() - t.min() < 1e-9:
        return 0.0
    slope_per_s = np.polyfit(t - t.mean(), y, 1)[0]
    return float(slope_per_s * 60.0)


def state_from_history(row: dict, history: list) -> dict:
    """Everything the model needs about *now*, built from measured values only."""
    return {
        "t_in": row["t_in"],
        "t_wall": row["t_wall"],
        "t_out": row["t_out"],
        "heater_w": row["heater_w"],
        "slope_in": slope_per_min(history, "t_in"),
        "slope_out": slope_per_min(history, "t_out"),
    }


def featurise_session(df: pd.DataFrame) -> pd.DataFrame:
    """Add the nine inputs, the label, and the action-stability flag to one session.

    The label is how much the measured indoor temperature changes over the
    next 10 minutes. Measured, not true, because that is what a real
    controller would be scored against.

    `action_stable` marks the rows where u and f stayed put for the whole
    10-minute window. PLAN section 5 keeps only those, because otherwise the
    outcome is a blend of two different actions and the row teaches the model
    something that never happened.
    """
    df = df.copy().reset_index(drop=True)
    step = int(round(cfg.DT_CONTROL))
    horizon_rows = int(round(cfg.HORIZON_S / step))

    # Trends over the last 5 minutes, from the measured series.
    win = int(round(cfg.SLOPE_WINDOW_S / step))
    for src, dst in (("t_in", "slope_in"), ("t_out", "slope_out")):
        s = df[src]
        # Least-squares slope over a rolling window, in degC per minute.
        x = np.arange(win + 1, dtype=float) * step
        xc = x - x.mean()
        denom = float((xc ** 2).sum())
        df[dst] = (
            s.rolling(win + 1)
             .apply(lambda y, xc=xc, denom=denom: float(np.dot(xc, y) / denom), raw=True)
             * 60.0
        )
    df[["slope_in", "slope_out"]] = df[["slope_in", "slope_out"]].fillna(0.0)

    cols = build_features(
        df["t_in"].to_numpy(), df["t_wall"].to_numpy(), df["t_out"].to_numpy(),
        df["heater_w"].to_numpy(), df["slope_in"].to_numpy(), df["slope_out"].to_numpy(),
        df["u"].to_numpy(), df["f"].to_numpy(),
    )
    for name, col in zip(FEATURE_NAMES, cols):
        df[name] = col

    # Label: change in measured indoor temperature over the next 10 minutes.
    df["y_dt10"] = df["t_in"].shift(-horizon_rows) - df["t_in"]

    # Did the action hold still for the whole horizon?
    u_same = np.ones(len(df), dtype=bool)
    for k in range(1, horizon_rows + 1):
        u_same &= (df["u"].shift(-k) == df["u"]).fillna(False).to_numpy()
        u_same &= (df["f"].shift(-k) == df["f"]).fillna(False).to_numpy()
    # Also need a full 5 minutes of history behind the row for the trends.
    warm = np.asarray(df.index >= win)
    df["action_stable"] = u_same & warm & df["y_dt10"].notna().to_numpy()
    return df
