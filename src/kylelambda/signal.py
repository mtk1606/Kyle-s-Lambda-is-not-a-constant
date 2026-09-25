"""Phase 3: does forecast lambda predict the adverse selection that passive fills will suffer?

For a forecast made at the end of window t for window t+h, the thing to predict is
AS_{t+h}(tau): the notional-weighted maker markout at horizon tau on every fill in
window t+h (markout.py). Kyle's lambda is the market maker's price response per unit
of flow; if the forecast of it carries information, it should rank upcoming windows
by how badly the makers in them get picked off.

Two ICs, because they answer different questions:

    daily IC   Spearman rank correlation computed within each day, then averaged.
               Only intraday timing counts; day-level effects (a volatile day has both
               high lambda and high AS) are removed. t-stat is Newey-West over days.
    pooled IC  one Spearman correlation over all windows; includes day-level effects.

Predictors are put on a level scale before ranking (z forecast times the trailing
7-day median lambda known at time t) so the pooled IC is meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .forecast import DAY, to_grid
from .io import DATA, load_windows
from .stats import nw_var_mean


def load_panel(sym: str, est: str, W: int, h: int) -> pd.DataFrame:
    """Forecasts at t aligned with realized markouts and lambda in window t+h."""
    preds = pd.read_parquet(DATA / "preds" / sym / f"{est}_W{W}_h{h}.parquet")
    g = to_grid(load_windows(sym, 1, W), W)
    mk = pd.read_parquet(DATA / "markouts" / sym / f"W{W}.parquet").drop_duplicates("t0").set_index("t0")
    mk = mk.reindex(g.index)
    per_day = DAY // W
    scale = g[est].shift(1).rolling(7 * per_day, min_periods=7 * per_day // 3).median()
    p = pd.DataFrame(index=preds.index)
    for c in ["har", "trailing", "seasonal", "last"]:
        p[f"f_{c}"] = preds[c] * scale.reindex(preds.index)
    # classic toxicity proxies known at t, for comparison
    p["f_rv"] = g["rv_bps2"].reindex(p.index)
    p["f_abs_oi"] = (g["oi_usd"].abs() / g["notional_musd"]).rolling(3, min_periods=1).mean().reindex(p.index)
    p["f_amihud"] = g["amihud"].rolling(3, min_periods=1).mean().reindex(p.index)
    shift = lambda s: s.shift(-h).reindex(p.index)  # noqa: E731
    for tau in [1, 10, 60, 300]:
        p[f"as_{tau}"] = shift(mk[f"as_{tau}"])
        p[f"hs_{tau}"] = shift(mk[f"hs_{tau}"])
    p["notional_next"] = shift(g["notional_musd"])
    p["lambda_next"] = shift(g[est])
    p["date"] = pd.to_datetime(p.index + h * W, unit="s").strftime("%Y-%m-%d")
    return p


def daily_ic(p: pd.DataFrame, f: str, y: str, min_obs: int = 20) -> pd.Series:
    def one(d):
        d = d[[f, y]].dropna()
        if len(d) < min_obs:
            return np.nan
        return stats.spearmanr(d[f], d[y]).statistic
    return p.groupby("date").apply(one).dropna()


def ic_summary(p: pd.DataFrame, f: str, y: str) -> dict:
    ics = daily_ic(p, f, y)
    d = p[[f, y]].dropna()
    se = np.sqrt(nw_var_mean(ics.to_numpy(), lags=5))
    return {
        "ic_daily": ics.mean(),
        "ic_daily_t": ics.mean() / se,
        "ic_daily_pos_frac": (ics > 0).mean(),
        "n_days": len(ics),
        "ic_pooled": stats.spearmanr(d[f], d[y]).statistic,
        "n_windows": len(d),
    }


def incremental_regression(p: pd.DataFrame, y: str, cols: list[str], lags: int) -> pd.DataFrame:
    """AS on standardized, rank-gaussianized predictors, with Newey-West t-stats.

    Ranks (per day) make this robust to the fat tails in both lambda and AS, and
    per-day demeaning restricts it to the intraday question, like the daily IC.
    """
    d = p[cols + [y, "date"]].dropna().copy()
    for c in cols + [y]:
        r = d.groupby("date")[c].rank(pct=True)
        n = d.groupby("date")[c].transform("count")
        d[c] = stats.norm.ppf((r * n - 0.5) / n)
    X = np.c_[np.ones(len(d)), d[cols].to_numpy()]
    yy = d[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ beta
    # NW sandwich
    XtXi = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return pd.DataFrame({"coef": beta, "t": beta / np.sqrt(np.diag(V))}, index=["const"] + cols)
