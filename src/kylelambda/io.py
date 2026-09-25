"""Paths and loaders shared by the phase scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path("data")
RESULTS = Path("results")
FIG = RESULTS / "figures"


def bar_files(symbol: str) -> list[Path]:
    return sorted((DATA / "bars" / symbol).glob("????-??-??.parquet"))


def load_bars(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def window_path(symbol: str, bar_sec: int, window_sec: int) -> Path:
    return DATA / "windows" / symbol / f"k{bar_sec}_W{window_sec}.parquet"


def load_windows(symbol: str, bar_sec: int, window_sec: int) -> pd.DataFrame:
    df = pd.read_parquet(window_path(symbol, bar_sec, window_sec))
    df["ts"] = pd.to_datetime(df["t0"], unit="s")
    return df
