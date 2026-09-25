"""The anchor: Kyle (1985), solved numerically, plus a synthetic market with a known lambda path.

Sequential-auction equilibrium (Kyle 1985, Theorem 2). With N auctions, interval dt,
noise-trade variance sigma_u^2 per unit time, and prior variance Sigma_0 of the asset
value, the linear equilibrium satisfies, for n = 1..N,

    Sigma_n        = (1 - beta_n lambda_n dt) Sigma_{n-1}
    lambda_n       = beta_n Sigma_n / sigma_u^2
    beta_n dt      = (1 - 2 alpha_n lambda_n) / (2 lambda_n (1 - alpha_n lambda_n))
    alpha_{n-1}    = 1 / (4 lambda_n (1 - alpha_n lambda_n))
    alpha_N        = 0,     second-order condition lambda_n (1 - alpha_n lambda_n) > 0.

Substituting the second equation into the third gives a cubic in lambda_n given
(Sigma_n, alpha_n), so the system can be solved backward from a guess of Sigma_N and
shot forward to match Sigma_0. As N grows, lambda_n flattens to sigma_v / sigma_u:
the constant depth that this whole project argues with.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KyleEquilibrium:
    lam: np.ndarray    # lambda_n, n = 1..N
    beta: np.ndarray   # beta_n (trading intensity per unit time)
    sigma: np.ndarray  # Sigma_n, n = 0..N


def _lambda_root(Sn: float, alpha: float, su2: float, dt: float) -> float:
    # -2 a c lam^3 + 2 c lam^2 + 2 a lam - 1 = 0,  c = su2 dt / Sn
    c = su2 * dt / Sn
    roots = np.roots([-2 * alpha * c, 2 * c, 2 * alpha, -1.0]) if alpha > 0 else np.roots([2 * c, 0.0, -1.0])
    roots = roots[np.isreal(roots)].real
    ok = roots[(roots > 0) & (roots * (1 - alpha * roots) > 0) & (1 - 2 * alpha * roots > 0)]
    if len(ok) == 0:
        raise ValueError("no root satisfies the second-order condition")
    return float(ok.min())


def _backward(SN: float, N: int, su2: float, dt: float):
    lam = np.empty(N); beta = np.empty(N); sig = np.empty(N + 1)
    sig[N] = SN
    alpha = 0.0
    for n in range(N - 1, -1, -1):
        L = _lambda_root(sig[n + 1], alpha, su2, dt)
        b = L * su2 / sig[n + 1]
        lam[n], beta[n] = L, b
        sig[n] = sig[n + 1] / (1 - b * L * dt)
        alpha = 1 / (4 * L * (1 - alpha * L))
    return lam, beta, sig


def kyle_equilibrium(N: int, sigma_v: float = 1.0, sigma_u: float = 1.0, T: float = 1.0) -> KyleEquilibrium:
    su2, dt, S0 = sigma_u ** 2, T / N, sigma_v ** 2
    lo, hi = np.log(S0) - 30, np.log(S0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        try:
            *_, sig = _backward(np.exp(mid), N, su2, dt)
            too_big = sig[0] > S0
        except (ValueError, FloatingPointError, ZeroDivisionError):
            too_big = True
        if too_big:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-13:
            break
    lam, beta, sig = _backward(np.exp(lo), N, su2, dt)
    return KyleEquilibrium(lam, beta, sig)


def simulate_market(n_windows: int, bars_per_window: int, lam_path: np.ndarray, noise_bps: float = 1.0,
                    flow_df: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Bars with a known window-level lambda: r = lambda_w * x + e, x Student-t (fat tails)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(flow_df, size=(n_windows, bars_per_window))
    e = noise_bps * rng.standard_normal((n_windows, bars_per_window))
    r = lam_path[:, None] * x + e
    return x, r
