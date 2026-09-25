"""Window-level price-impact estimators.

All of them are variants of the regression practitioners actually run,

    r_i = a + lambda * x_i + e_i,      i = bars inside one estimation window,

with r in bps of the mid and x a signed-flow measure. They differ in what x is:

    kyle_usd   signed notional, $M                  (Kyle 1985; the textbook lambda)
    kyle_base  signed base quantity, coins          (same thing in native units)
    sqrt_usd   sum over taker orders of sign*sqrt($) (Hasbrouck 2009 functional form)
    count      #buy orders - #sell orders           (order-count imbalance; robust to size outliers)
    ofi        best-quote order-flow imbalance, $M  (Cont, Kukanov & Stoikov 2014; book venues only)

and an unsigned proxy that needs no trade signing,

    amihud     |r_window| / notional $M             (Amihud 2002, applied intraday)

Windows are contiguous, equal-length blocks of the day, so I reshape to
(n_windows, bars_per_window) and do the OLS with sufficient statistics. Each window also
gets two half-sample estimates from interleaved bars (odd and even); their disagreement
is a model-free measure of the estimator's own sampling noise, used in audit.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import resample

FLOWS = {
    "kyle_usd": lambda b: b["sn"].to_numpy() / 1e6,
    "kyle_base": lambda b: b["sv"].to_numpy(),
    "sqrt_usd": lambda b: b["ssqrt_usd"].to_numpy(),
    "count": lambda b: (b["n_buy"] - b["n_sell"]).to_numpy(),
    "ofi": lambda b: b["ofi_usd"].to_numpy() / 1e6,
}


def _ols_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise weighted OLS slope with intercept; w is a 0/1 validity mask. Returns (slope, white_se)."""
    n = w.sum(1)
    xm = (w * x).sum(1) / np.maximum(n, 1)
    ym = (w * y).sum(1) / np.maximum(n, 1)
    xc = (x - xm[:, None]) * w
    yc = (y - ym[:, None]) * w
    sxx = (xc * xc).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (xc * yc).sum(1) / sxx
        e = (yc - b[:, None] * xc) * w
        se = np.sqrt((xc * xc * e * e).sum(1)) / sxx  # HC0; bars are ~uncorrelated at the sizes used
    bad = (n < 4) | (sxx <= 0)
    b[bad] = np.nan
    se[bad] = np.nan
    return b, se


def valid_seconds(bars1s: pd.DataFrame, max_gap: int = 300) -> np.ndarray:
    """False inside tape outages: runs of more than max_gap seconds with no trades at all."""
    active = bars1s["n_fills"].to_numpy() > 0
    idx = np.flatnonzero(active)
    ok = np.ones(len(active), dtype=bool)
    if len(idx) == 0:
        return ~ok
    edges = np.r_[-1, idx, len(active)]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a - 1 > max_gap:
            ok[a + 1:b] = False
    return ok


def window_lambdas(bars1s: pd.DataFrame, bar_sec: int, window_sec: int,
                   flows: tuple[str, ...] = ("kyle_usd", "sqrt_usd", "count"),
                   max_invalid: float = 0.2, ok1: np.ndarray | None = None) -> pd.DataFrame:
    """Estimate lambda on every window of one day. One row per window.

    ok1 marks usable seconds; by default, anything outside a tape outage. Book-level
    venues pass their own mask (a quiet tape is not an outage when quotes still update).
    """
    assert window_sec % bar_sec == 0
    per = window_sec // bar_sec
    if ok1 is None:
        ok1 = valid_seconds(bars1s)
    ok1 = ok1[: len(bars1s) // bar_sec * bar_sec]
    b = resample(bars1s, bar_sec)
    okb = ok1.reshape(-1, bar_sec).all(1).astype(float)
    nwin = len(b) // per
    b = b.iloc[: nwin * per]
    okb = okb[: nwin * per].reshape(nwin, per)
    # the first bar of the day has no previous mid; drop it
    okb[0, 0] = 0.0
    y = b["ret_bps"].to_numpy().reshape(nwin, per)
    odd = np.zeros(per)
    odd[1::2] = 1.0
    out = {
        "t0": b.index.to_numpy()[::per],
        "valid_frac": okb.mean(1),
        "notional_musd": b["notional"].to_numpy().reshape(nwin, per).sum(1) / 1e6,
        "n_orders": b["n_orders"].to_numpy().reshape(nwin, per).sum(1),
        "rv_bps2": (y ** 2 * okb).sum(1),
        "ret_bps": (y * okb).sum(1),
        "oi_usd": b["sn"].to_numpy().reshape(nwin, per).sum(1) / 1e6,
        "spread_bps": 1e4 * (b["spread_hat"] / b["mid"]).to_numpy().reshape(nwin, per).mean(1),
    }
    for f in flows:
        x = FLOWS[f](b).reshape(nwin, per).astype(float)
        lam, se = _ols_slope(x, y, okb)
        lo, _ = _ols_slope(x, y, okb * odd)
        le, _ = _ols_slope(x, y, okb * (1 - odd))
        out[f] = lam
        out[f + "_se"] = se
        out[f + "_odd"] = lo
        out[f + "_even"] = le
    with np.errstate(divide="ignore", invalid="ignore"):
        out["amihud"] = np.abs(out["ret_bps"]) / out["notional_musd"]
    df = pd.DataFrame(out)
    bad = df["valid_frac"] < 1 - max_invalid
    df.loc[bad, [c for c in df.columns if c not in ("t0", "valid_frac")]] = np.nan
    return df
