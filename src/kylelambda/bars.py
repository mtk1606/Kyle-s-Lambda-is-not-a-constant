"""One-second bars on a full calendar grid.

Everything downstream (lambda estimators, markouts, forecasts) is built from these, so
the one design decision that matters here is the price series. Regressing trade-price
changes on signed flow mechanically picks up bid-ask bounce: a buy prints at the ask,
a sell at the bid, and the "impact" you estimate is partly just the spread. For venues
where I have the book (Coinbase L3, LOBSTER) I use the true mid. For the Binance trade
tape I use a bounce-corrected proxy,

    m_i = p_i - sign_i * s_hat / 2,

where s_hat is the hourly median absolute price jump between consecutive opposite-signed
trades less than a second apart (the Roll-style bounce size).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400

BAR_COLUMNS = [
    "n_fills", "n_orders", "n_buy", "n_sell", "vol", "notional",
    "sv", "sn", "ssqrt", "ssqrt_usd", "mid", "last_px", "spread_hat",
]


def bounce_spread(time: np.ndarray, price: np.ndarray, sign: np.ndarray, bucket: int = 3600) -> np.ndarray:
    """Per-trade estimate of the quoted spread from opposite-signed consecutive trades."""
    dt = np.diff(time)
    dp = np.abs(np.diff(price))
    flip = (sign[1:] != sign[:-1]) & (dt < 1.0) & (dp > 0)
    b = (time[1:] // bucket).astype(np.int64)
    s = pd.Series(dp[flip]).groupby(b[flip]).median()
    tb = (time // bucket).astype(np.int64)
    out = s.reindex(tb).to_numpy()
    # hours with no qualifying flips inherit the day median, then the smallest non-zero jump
    fallback = np.nanmedian(dp[flip]) if flip.any() else np.nanmin(dp[dp > 0]) if (dp > 0).any() else 0.0
    return np.where(np.isfinite(out), out, fallback)


def build_second_bars(trades: pd.DataFrame, orders: pd.DataFrame, day_start: int) -> pd.DataFrame:
    """Aggregate a day of trades (and their parent taker orders) onto an 86,400-row grid."""
    t0, t1 = day_start, day_start + DAY
    tr = trades[(trades["time"] >= t0) & (trades["time"] < t1)]
    od = orders[(orders["time"] >= t0) & (orders["time"] < t1)]

    time = tr["time"].to_numpy()
    px = tr["price"].to_numpy()
    sg = tr["sign"].to_numpy().astype(np.float64)
    qty = tr["qty"].to_numpy()
    s_hat = bounce_spread(time, px, sg)
    mid_tick = px - sg * s_hat / 2.0

    sec = (time - t0).astype(np.int64)
    osec = (od["time"].to_numpy() - t0).astype(np.int64)
    osg = od["sign"].to_numpy().astype(np.float64)

    def acc(idx, w):
        return np.bincount(idx, weights=w, minlength=DAY)

    bars = pd.DataFrame({
        "n_fills": np.bincount(sec, minlength=DAY),
        "n_orders": np.bincount(osec, minlength=DAY),
        "n_buy": acc(osec, (osg > 0).astype(float)),
        "n_sell": acc(osec, (osg < 0).astype(float)),
        "vol": acc(sec, qty),
        "notional": acc(sec, px * qty),
        "sv": acc(sec, sg * qty),
        "sn": acc(sec, sg * px * qty),
        "ssqrt": acc(osec, osg * np.sqrt(od["qty"].to_numpy())),
        "ssqrt_usd": acc(osec, osg * np.sqrt(od["notional"].to_numpy())),
    })
    # end-of-second values: last trade in each second, forward filled across quiet seconds
    last_idx = pd.Series(np.arange(len(sec))).groupby(sec).max()
    mid = np.full(DAY, np.nan)
    last = np.full(DAY, np.nan)
    sh = np.full(DAY, np.nan)
    mid[last_idx.index] = mid_tick[last_idx.to_numpy()]
    last[last_idx.index] = px[last_idx.to_numpy()]
    sh[last_idx.index] = s_hat[last_idx.to_numpy()]
    bars["mid"] = pd.Series(mid).ffill().bfill().to_numpy()
    bars["last_px"] = pd.Series(last).ffill().bfill().to_numpy()
    bars["spread_hat"] = pd.Series(sh).ffill().bfill().to_numpy()
    bars.index = pd.RangeIndex(t0, t1, name="t")
    for c in ["n_fills", "n_orders"]:
        bars[c] = bars[c].astype(np.int32)
    return bars


def resample(bars: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Aggregate 1s bars to k-second bars. Flow columns sum, mid takes the last value."""
    if seconds == 1:
        out = bars.copy()
    else:
        key = (bars.index.to_numpy() - bars.index[0]) // seconds
        g = bars.groupby(key)
        flow = ["n_fills", "n_orders", "n_buy", "n_sell", "vol", "notional", "sv", "sn", "ssqrt", "ssqrt_usd"]
        flow += [c for c in ("ofi", "ofi_usd", "n_quotes") if c in bars]  # book-level venues only
        sums = g[flow].sum()
        last = g[["mid", "last_px", "spread_hat"]].last()
        out = pd.concat([sums, last], axis=1)
        out.index = bars.index[0] + out.index.to_numpy() * seconds
        out.index.name = "t"
    prev = np.r_[bars["mid"].iloc[0], out["mid"].to_numpy()[:-1]]
    out["ret_bps"] = 1e4 * np.log(out["mid"].to_numpy() / prev)
    return out
