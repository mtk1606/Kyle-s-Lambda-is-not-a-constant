import numpy as np
import pandas as pd

from kylelambda.forecast import seasonal_forecast, walk_forward
from kylelambda.policy import causal_threshold


def _grid(n_days=120, W=3600, seed=0):
    rng = np.random.default_rng(seed)
    per = 86400 // W
    n = n_days * per
    t0 = 1_500_000_000 - 1_500_000_000 % 86400
    idx = t0 + W * np.arange(n)
    # persistent AR(1) lambda with an intraday cycle, observed with noise
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + 0.3 * rng.standard_normal()
    season = 0.3 * np.sin(2 * np.pi * (np.arange(n) % per) / per)
    lam = np.exp(ar + season)
    obs = lam * (1 + 0.3 * rng.standard_normal(n))
    g = pd.DataFrame({"kyle_usd": obs, "sqrt_usd": obs, "count": obs, "rv_bps2": lam * 100,
                      "notional_musd": 1 / lam, "n_orders": 100.0, "spread_bps": 1.0,
                      "oi_usd": rng.standard_normal(n) * 0.1, "ret_bps": rng.standard_normal(n)}, index=idx)
    return g


def test_no_lookahead_in_seasonal():
    g = _grid()
    z = g["kyle_usd"]
    s1 = seasonal_forecast(z, 3600, 1)
    z2 = z.copy(); z2.iloc[2000:] = 1e6  # scramble the future
    s2 = seasonal_forecast(z2, 3600, 1)
    assert np.allclose(s1.iloc[:2000].fillna(-9), s2.iloc[:2000].fillna(-9))


def test_walk_forward_no_lookahead_and_beats_constant():
    g = _grid()
    p1 = walk_forward(g, "kyle_usd", 3600, 1, burn_days=30)
    g2 = g.copy(); g2.iloc[-500:, :] = g2.iloc[-500:, :] * 50
    p2 = walk_forward(g2, "kyle_usd", 3600, 1, burn_days=30)
    # a forecast for an early month must not change when late data changes
    early = p1.index < g.index[-24 * 35]
    assert np.allclose(p1.loc[early, "har"].fillna(0), p2.loc[early, "har"].fillna(0))
    d = p1.dropna()
    mse = lambda f: np.mean((d["target"] - d[f]) ** 2)  # noqa: E731
    assert mse("har") < mse("constant") and mse("har") < mse("last")


def test_threshold_is_causal():
    s = pd.Series(np.arange(5000, dtype=float))
    thr = causal_threshold(s, 3600, 0.8)
    s2 = s.copy(); s2.iloc[3000] = 1e9
    thr2 = causal_threshold(s2, 3600, 0.8)
    assert np.allclose(thr.iloc[:3001].fillna(-1), thr2.iloc[:3001].fillna(-1))
