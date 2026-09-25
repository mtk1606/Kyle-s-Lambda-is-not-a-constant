"""Phase 1: how much of an estimated lambda is lambda, and how much is estimator noise.

The core quantity is split-half reliability. Every window is estimated twice from
interleaved bars (odd, even). If both halves measure the same true lambda_w plus
independent noise, then across windows

    corr(lambda_odd, lambda_even) = Var(lambda_w) / (Var(lambda_w) + Var(noise_half)),

and Spearman-Brown converts that into the reliability of the full-window estimate,
rho = 2r / (1 + r): the share of the observed cross-window variance that is real.
rho near 0 means a rolling lambda at that setting is mostly noise; rho near 1 means
its movements are real.

Separately, (lambda_odd - lambda_even)^2 / 4 is an unbiased, model-free estimate of the
full-window sampling variance, which gives the per-window relative noise
sqrt(noise) / |median lambda| without trusting any standard-error formula.

Everything is winsorized at the 1st/99th percentile first. Order flow is fat-tailed
and a single 5-minute window with a whale print would otherwise dominate the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def winsor(x: np.ndarray, q: float = 0.01) -> np.ndarray:
    lo, hi = np.nanquantile(x, [q, 1 - q])
    return np.clip(x, lo, hi)


def reliability(df: pd.DataFrame, col: str, q: float = 0.01, min_n: int = 30) -> dict:
    d = df[[col, col + "_odd", col + "_even"]].dropna()
    n = len(d)
    if n < min_n:
        return {"n_windows": n}
    full = winsor(d[col].to_numpy(), q)
    o = winsor(d[col + "_odd"].to_numpy(), q)
    e = winsor(d[col + "_even"].to_numpy(), q)
    r = np.corrcoef(o, e)[0, 1]
    rs = stats.spearmanr(o, e).statistic
    noise_var = np.mean((o - e) ** 2) / 4.0
    med = np.median(full)
    return {
        "n_windows": n,
        "median_lambda": med,
        "iqr_lambda": np.subtract(*np.quantile(full, [0.75, 0.25])),
        "frac_negative": float((d[col] < 0).mean()),
        "split_half_r": r,
        "reliability_sb": 2 * r / (1 + r) if r > -1 else np.nan,
        "reliability_spearman_sb": 2 * rs / (1 + rs) if rs > -1 else np.nan,
        "reliability_noise": 1 - noise_var / np.var(full),
        "rel_noise": np.sqrt(noise_var) / abs(med) if med != 0 else np.nan,
    }


def permutation_constancy(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int = 200,
                          rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Permutation test of H0: one lambda for all groups (e.g. the 24 hours of a day).

    Statistic: the precision-weighted dispersion of per-group slopes around the pooled
    slope, i.e. the Chow statistic without its normality assumption. Under H0 the
    (flow, return) pairs are exchangeable across groups, so shuffling the group labels
    gives the null distribution; heteroskedasticity travels with each pair.
    """
    rng = rng or np.random.default_rng(0)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[m], y[m], groups[m]
    ug, gi = np.unique(groups, return_inverse=True)
    G = len(ug)

    def stat(g):
        n = np.bincount(g, minlength=G).astype(float)
        sx = np.bincount(g, x, G); sy = np.bincount(g, y, G)
        sxx = np.bincount(g, x * x, G); sxy = np.bincount(g, x * y, G)
        with np.errstate(invalid="ignore", divide="ignore"):
            vxx = sxx - sx * sx / n
            b = (sxy - sx * sy / n) / vxx
        ok = (n > 3) & (vxx > 0)
        bp = np.sum((sxy - sx * sy / n)[ok]) / np.sum(vxx[ok])
        return np.sum(vxx[ok] * (b[ok] - bp) ** 2)

    s0 = stat(gi)
    null = np.array([stat(rng.permutation(gi)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= s0)) / (n_perm + 1)
    return float(s0 / np.mean(null)), float(p)
