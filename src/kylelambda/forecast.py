"""Phase 2: lambda as a time series.

Target. The per-window estimate lambda_hat_t, divided by its own median over the
previous 7 days. The normalization removes the slow drift in dollar depth (BTC traded
between $3k and $13k in this sample, so bps-per-$M moves a lot for boring reasons) and
leaves the intraday and day-to-day dynamics I care about. z_t = 1 is "lambda at its
usual level", which is exactly the Kyle-constant forecast.

Forecasters, all strictly causal:

    constant   z = 1                                   Kyle: depth is constant
    last       the previous window's estimate           a short rolling lambda, used raw
    trailing   EWMA of past z, half-life tuned in-sample the desk practice: a smoothed rolling lambda
    seasonal   mean z around the same time of day        deterministic intraday pattern
               (weekday/weekend separately), past 28 days
    *_cal      a + b * (trailing or seasonal), a and b   the same baselines, optimally scaled,
               fitted on the training window              so the comparison is not a strawman
    har        OLS on HAR lags of z, the seasonal term,   conditional model
               and lagged market-state variables

Evaluation is walk-forward by calendar month with a one-day embargo between the end of
the training data and the first forecast. Targets are winsorized at training-sample
quantiles so one whale print cannot decide the result.

The noise ceiling matters for reading R^2. lambda_hat = lambda + noise, and noise is
unpredictable by construction, so the best achievable R^2 against lambda_hat is roughly
the Phase 1 reliability. I report R^2 and R^2 / reliability ("share of the forecastable
variance captured").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86_400


def to_grid(win: pd.DataFrame, window_sec: int) -> pd.DataFrame:
    """Place per-day window tables on one continuous time grid (missing days become NaN)."""
    win = win.sort_values("t0").drop_duplicates("t0").set_index("t0")
    grid = np.arange(win.index.min(), win.index.max() + window_sec, window_sec)
    return win.reindex(grid)


def normalize(s: pd.Series, window_sec: int, days: int = 7) -> pd.Series:
    """Divide by the trailing `days` median, using only strictly past windows."""
    n = days * DAY // window_sec
    med = s.shift(1).rolling(n, min_periods=n // 3).median()
    return s / med


def seasonal_forecast(z: pd.Series, window_sec: int, h: int, days: int = 28,
                      smooth_sec: int = 1800) -> pd.Series:
    """Forecast of z_{t+h} from the mean of z around the same time of day over the past `days` days.

    Each past day contributes the average of z within +/- smooth_sec of the target
    slot (a single 5-minute slot averaged over 20 days is too noisy to be useful).
    Weekends get their own profile because crypto volume drops on Saturday and Sunday.
    A past day is only used if its whole smoothing band ends at or before t.
    """
    per_day = DAY // window_sec
    assert h <= per_day, "seasonal forecast only defined up to one day ahead"
    J = smooth_sec // window_sec
    n = len(z)
    t = z.index.to_numpy()
    wkend = pd.to_datetime(t + h * window_sec, unit="s").dayofweek >= 5
    zs = z.rolling(2 * J + 1, center=True, min_periods=J + 1).mean().to_numpy()
    lags = []
    for d in range(1, days + 1):
        if d * per_day < h + J:
            continue
        src = np.arange(n) + h - d * per_day
        v = np.where(src >= 0, zs[np.clip(src, 0, n - 1)], np.nan)
        src_wk = pd.to_datetime(t + (h - d * per_day) * window_sec, unit="s").dayofweek >= 5
        lags.append(np.where(src_wk == wkend, v, np.nan))
    L = np.vstack(lags)
    cnt = np.isfinite(L).sum(0)
    tot = np.nansum(L, axis=0)
    out = np.where(cnt >= 3, tot / np.maximum(cnt, 1), np.nan)
    return pd.Series(out, index=z.index)


def ewma_past(z: pd.Series, halflife: float) -> pd.Series:
    return z.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()


STATE_COLS = ["rv_bps2", "notional_musd", "n_orders", "spread_bps"]


def build_features(g: pd.DataFrame, est: str, window_sec: int, h: int) -> pd.DataFrame:
    """Feature matrix for forecasting z_{t+h} at the end of window t."""
    z = normalize(g[est], window_sec)
    per_day = DAY // window_sec
    X = pd.DataFrame(index=g.index)
    X["z"] = z
    for L in [max(2, per_day // 48), max(3, per_day // 12), max(4, per_day // 4), per_day]:
        X[f"z_mean{L}"] = z.rolling(L, min_periods=max(1, L // 2)).mean()
    X["seasonal"] = seasonal_forecast(z, window_sec, h)
    for alt in ["sqrt_usd", "count"]:
        if alt != est and alt in g:
            X[f"{alt}_z"] = normalize(g[alt], window_sec).rolling(3, min_periods=1).mean()
    for c in STATE_COLS:
        v = g[c].where(g[c] > 0)
        lv = np.log(v) - np.log(v.shift(1).rolling(7 * per_day, min_periods=per_day).median())
        X[f"log_{c}"] = lv
        X[f"log_{c}_mean"] = lv.rolling(max(3, per_day // 24), min_periods=1).mean()
    X["abs_oi_share"] = (g["oi_usd"].abs() / g["notional_musd"]).rolling(3, min_periods=1).mean()
    X["abs_ret"] = np.log1p(g["ret_bps"].abs())
    X["weekend"] = (pd.to_datetime(g.index + h * window_sec, unit="s").dayofweek >= 5).astype(float)
    X["target"] = z.shift(-h)
    return X


@dataclass
class WalkForwardResult:
    preds: pd.DataFrame          # index t, columns: target + each forecaster
    reliability: float | None = None


def _ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    Xa = np.c_[np.ones(len(X)), X]
    A = Xa.T @ Xa
    A[1:, 1:] += ridge * np.eye(X.shape[1]) * len(X)
    return np.linalg.solve(A, Xa.T @ y)


def walk_forward(g: pd.DataFrame, est: str, window_sec: int, h: int, burn_days: int = 60,
                 embargo_days: int = 1) -> pd.DataFrame:
    X = build_features(g, est, window_sec, h)
    feat = [c for c in X.columns if c != "target"]
    ts = pd.to_datetime(X.index, unit="s")
    months = ts.to_period("M")
    start = ts.min() + pd.Timedelta(days=burn_days)
    out = []
    z = X["z"]
    for m in months.unique():
        test = (months == m) & (ts >= start)
        if not test.any():
            continue
        cutoff = ts[test].min() - pd.Timedelta(days=embargo_days)
        # the target of a training row is observed h windows later, so it must end before the cutoff too
        train = (ts + pd.Timedelta(seconds=h * window_sec) < cutoff)
        tr = X[train].dropna()
        if len(tr) < 500:
            continue
        lo, hi = tr["target"].quantile([0.01, 0.99])
        y = tr["target"].clip(lo, hi).to_numpy()
        # feature winsorization bounds also come from training data only
        flo, fhi = tr[feat].quantile(0.005), tr[feat].quantile(0.995)
        Xtr = tr[feat].clip(flo, fhi, axis=1)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        beta = _ols(((Xtr - mu) / sd).to_numpy(), y)
        # trailing baseline: pick the EWMA half-life that minimises in-sample MSE
        best, best_hl = np.inf, 1.0
        trz = z[train]
        tgt = X["target"][train].clip(lo, hi)
        for hl in [1, 2, 3, 6, 12, 24, 48, 96, 288]:
            e = ewma_past(trz, hl)
            mse = np.nanmean((tgt - e) ** 2)
            if mse < best:
                best, best_hl = mse, hl
        te = X[test]
        Xte = te[feat].clip(flo, fhi, axis=1)
        ok = Xte.notna().all(axis=1) & te["target"].notna()
        pred_har = pd.Series(np.nan, index=te.index)
        pred_har[ok] = np.c_[np.ones(ok.sum()), ((Xte[ok] - mu) / sd).to_numpy()] @ beta
        ew_all = ewma_past(z[ts <= ts[test].max()], best_hl)
        ew = ew_all.reindex(te.index)
        # calibrated baselines: y = a + b * baseline, fitted on the same training rows,
        # so the conditional model is never compared against a strawman
        cal = {}
        for name, src_tr, src_te in [("trailing_cal", ew_all.reindex(tr.index), ew),
                                     ("seasonal_cal", tr["seasonal"], te["seasonal"])]:
            m = src_tr.notna()
            a, b = _ols(src_tr[m].to_numpy()[:, None], y[m.to_numpy()], ridge=0.0)
            cal[name] = a + b * src_te
        out.append(pd.DataFrame({
            "target": te["target"].clip(lo, hi),
            "target_raw": te["target"],
            "constant": 1.0,
            # raw baselines are clipped to the same training bounds as the target
            "last": te["z"].clip(lo, hi),
            "trailing": ew.clip(lo, hi),
            "seasonal": te["seasonal"].clip(lo, hi),
            "trailing_cal": cal["trailing_cal"],
            "seasonal_cal": cal["seasonal_cal"],
            "har": pred_har,
            "hl": best_hl,
        }))
    return pd.concat(out) if out else pd.DataFrame()


def oos_r2(y: np.ndarray, f: np.ndarray, bench: np.ndarray) -> float:
    return 1 - np.sum((y - f) ** 2) / np.sum((y - bench) ** 2)
