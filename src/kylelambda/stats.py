"""Inference helpers: HAC variance, Diebold-Mariano, block bootstrap."""
from __future__ import annotations

import numpy as np
from scipy import stats


def nw_var_mean(x: np.ndarray, lags: int) -> float:
    """Newey-West variance of the sample mean of x."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    u = x - x.mean()
    v = u @ u / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (u[L:] @ u[:-L]) / n
    return v / n


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int = 1, lags: int | None = None) -> tuple[float, float]:
    """DM test on squared-error loss. Positive stat means model a has *lower* loss than b."""
    d = np.asarray(e_b, float) ** 2 - np.asarray(e_a, float) ** 2
    d = d[np.isfinite(d)]
    lags = lags if lags is not None else max(h, int(np.ceil(len(d) ** (1 / 3))))
    se = np.sqrt(nw_var_mean(d, lags))
    t = d.mean() / se
    return float(t), float(2 * stats.norm.sf(abs(t)))


def block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0,
                         ci: float = 0.95) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of a serially dependent series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(1)
    a = (1 - ci) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def cluster_bootstrap(values_by_cluster: list[np.ndarray], stat, n_boot: int = 2000, seed: int = 0,
                      ci: float = 0.95) -> tuple[float, float, float]:
    """Resample whole clusters (days) with replacement and recompute stat(concatenated)."""
    rng = np.random.default_rng(seed)
    k = len(values_by_cluster)
    point = stat(np.concatenate(values_by_cluster))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        boots[b] = stat(np.concatenate([values_by_cluster[i] for i in pick]))
    a = (1 - ci) / 2
    return float(point), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
