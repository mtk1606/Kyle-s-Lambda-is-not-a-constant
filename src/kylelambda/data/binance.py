"""Binance spot raw trades (2018-04 .. 2019-11) from the Nucs/cryptocurrency-ticks-data mirror.

Each daily zip holds one CSV of raw (not aggregated) trades:

    Id, time, Price, Quantity, IsBuyerMaker, BuyerOrderId, SellerOrderId, IsBestPriceMatch

IsBuyerMaker == True means the resting order was the buy, so the aggressor sold.
The mirror's README is ambiguous on this, but it is Binance's documented convention
and it is the one that makes signed flow and price changes positively correlated
(checked in tests/test_binance_sign.py on a real day).

Timestamps are UTC epoch seconds. File names are shifted: the file labelled D starts
at 00:00 UTC of D-1, so I key everything off the timestamps and never off the name.

The mirror can be read either from a plain checkout (a directory of zips) or straight
out of a partial git clone, which saves ~5 GB of duplicated working-tree files.
"""
from __future__ import annotations

import io
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_COLS = ["id", "time", "price", "qty", "is_buyer_maker", "buyer_oid", "seller_oid", "best_match"]


@dataclass(frozen=True)
class BinanceMirror:
    root: Path  # repo root of the mirror (git clone) or a directory with data/<SYM>/*.zip

    def _is_git(self) -> bool:
        return (self.root / ".git").exists()

    def files(self, symbol: str) -> list[str]:
        if self._is_git():
            out = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD", f"data/{symbol}"],
                cwd=self.root, check=True, capture_output=True, text=True,
            ).stdout.split()
        else:
            out = [str(p.relative_to(self.root)) for p in (self.root / "data" / symbol).glob("*.zip")]
        return sorted(out)

    def read_bytes(self, relpath: str) -> bytes:
        if self._is_git():
            return subprocess.run(
                ["git", "cat-file", "blob", f"HEAD:{relpath}"],
                cwd=self.root, check=True, capture_output=True,
            ).stdout
        return (self.root / relpath).read_bytes()

    def load_day(self, relpath: str) -> pd.DataFrame:
        raw = self.read_bytes(relpath)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                df = pd.read_csv(
                    fh, header=0, names=_COLS,
                    # order ids read as float so a truncated last line (it happens) becomes NaN
                    dtype={"id": np.int64, "time": np.float64, "price": np.float64, "qty": np.float64,
                           "buyer_oid": np.float64, "seller_oid": np.float64},
                    converters={"is_buyer_maker": _to_bool, "best_match": _to_bool},
                )
        return clean_trades(df)


def _to_bool(x: str) -> bool:
    return x.strip().lower() == "true"


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicate ids and non-positive prints, and attach aggressor side and taker id."""
    df = df.dropna(subset=["time", "price", "qty", "buyer_oid", "seller_oid"])
    df = df[(df["price"] > 0) & (df["qty"] > 0)].copy()
    df["buyer_oid"] = df["buyer_oid"].astype(np.int64)
    df["seller_oid"] = df["seller_oid"].astype(np.int64)
    df = df.drop_duplicates("id").sort_values("id", kind="stable").reset_index(drop=True)
    buyer_maker = df["is_buyer_maker"].to_numpy(bool)
    df["sign"] = np.where(buyer_maker, -1, 1).astype(np.int8)
    df["taker_oid"] = np.where(buyer_maker, df["seller_oid"], df["buyer_oid"])
    return df


def taker_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse fills into parent taker orders.

    A single marketable order that sweeps several resting orders shows up as several
    trades sharing a taker order id. Kyle's model is written in terms of the order, not
    the fill, and the sqrt-volume estimator is only meaningful at the order level.
    """
    brk = (trades["taker_oid"].to_numpy() != np.roll(trades["taker_oid"].to_numpy(), 1))
    brk[0] = True
    grp = np.cumsum(brk) - 1
    g = trades.groupby(grp, sort=False)
    out = pd.DataFrame({
        "time": g["time"].first().to_numpy(),
        "sign": g["sign"].first().to_numpy(),
        "qty": g["qty"].sum().to_numpy(),
        "notional": (trades["price"] * trades["qty"]).groupby(grp, sort=False).sum().to_numpy(),
        "first_px": g["price"].first().to_numpy(),
        "last_px": g["price"].last().to_numpy(),
        "n_fills": g.size().to_numpy(),
    })
    return out
