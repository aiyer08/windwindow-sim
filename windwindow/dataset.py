"""
Step 9: loading the logs and splitting them safely.

The one thing that must not go wrong here is the split. Rows are 30 seconds
apart, so a row and its neighbour are almost the same measurement. If the
split is made row by row at random, nearly every test row has a near-duplicate
sitting in the training set, and the model scores brilliantly on the test set
while having learned nothing that generalises. Splitting by whole session is
the fix, and `assert_no_leakage` checks it actually held.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from .features import FEATURE_NAMES, featurise_session

TRAIN_DIR = pathlib.Path("data/train")


def load_sessions(directory: pathlib.Path = TRAIN_DIR) -> pd.DataFrame:
    """Read every session CSV, add features and labels, and stack them.

    Featurising happens per session. It has to: the trends and the label look
    forward and backward in time, and running them across a concatenated frame
    would blend the end of one session into the start of the next.
    """
    files = sorted(pathlib.Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no session CSVs in {directory}; run step08_collect.py")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(featurise_session(df))
    return pd.concat(frames, ignore_index=True)


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Only the rows whose action held still for the whole 10-minute horizon."""
    return df[df["action_stable"]].reset_index(drop=True)


def split(df: pd.DataFrame):
    """Separate training and held-out sessions. Returns (train, test)."""
    d = usable(df)
    return (d[d["split"] == "train"].reset_index(drop=True),
            d[d["split"] == "test"].reset_index(drop=True))


def xy(df: pd.DataFrame):
    """Feature matrix and label vector."""
    return df[FEATURE_NAMES].to_numpy(dtype=float), df["y_dt10"].to_numpy(dtype=float)


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> set:
    """Fail loudly if any session appears on both sides of the split."""
    a = set(train["session_id"].unique())
    b = set(test["session_id"].unique())
    overlap = a & b
    if overlap:
        raise AssertionError(f"{len(overlap)} sessions appear in both splits: {sorted(overlap)[:5]}")
    return overlap
