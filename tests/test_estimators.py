import numpy as np
import pandas as pd

from kylelambda.audit import permutation_constancy, reliability
from kylelambda.estimators import _ols_slope, valid_seconds
from kylelambda.kyle import simulate_market


def test_ols_slope_recovers_known_lambda():
    lam = np.array([0.5, 1.0, 2.0])
    x, r = simulate_market(3, 20_000, lam, noise_bps=1.0, seed=1)
    b, se = _ols_slope(x, r, np.ones_like(x))
    assert np.allclose(b, lam, atol=5 * se.max())


def test_mask_excludes_bars():
    x = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
    y = 2 * x
    y[0, -1] = -1e6  # a corrupted bar that the mask must remove
    w = np.array([[1, 1, 1, 1, 0.0]])
    b, _ = _ols_slope(x, y, w)
    assert np.isclose(b[0], 2.0)


def _split_half_frame(lam, per=200, noise=1.0, seed=0):
    x, r = simulate_market(len(lam), per, lam, noise_bps=noise, seed=seed)
    odd = np.zeros(per); odd[1::2] = 1
    one = np.ones_like(x)
    return pd.DataFrame({
        "est": _ols_slope(x, r, one)[0],
        "est_odd": _ols_slope(x, r, one * odd)[0],
        "est_even": _ols_slope(x, r, one * (1 - odd))[0],
    })


def test_reliability_matches_truth():
    # true lambda varies across windows; reliability should be Var(true) / Var(estimate)
    rng = np.random.default_rng(3)
    lam = 1 + 0.1 * rng.standard_normal(4000)
    df = _split_half_frame(lam, per=200, noise=1.0, seed=4)
    out = reliability(df, "est", q=0.0)
    truth = np.var(lam) / np.var(df["est"])
    assert abs(out["reliability_sb"] - truth) < 0.06
    assert abs(out["reliability_noise"] - truth) < 0.06


def test_reliability_zero_when_lambda_constant():
    df = _split_half_frame(np.ones(4000), per=100, noise=1.0, seed=5)
    assert abs(reliability(df, "est", q=0.0)["reliability_sb"]) < 0.08


def test_permutation_test_size_and_power():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(24), 300)
    # size: constant lambda, heteroskedastic noise across hours -> rejections near nominal
    rej = 0
    for i in range(60):
        x = rng.standard_t(3, len(groups))
        e = rng.standard_normal(len(groups)) * (0.5 + groups / 24)
        _, p = permutation_constancy(x, 1.0 * x + e, groups, n_perm=99, rng=rng)
        rej += p <= 0.05
    assert rej / 60 < 0.15
    # power: lambda doubles in the second half of the day
    x = rng.standard_t(3, len(groups))
    lam = np.where(groups < 12, 1.0, 2.0)
    _, p = permutation_constancy(x, lam * x + rng.standard_normal(len(groups)), groups, n_perm=99, rng=rng)
    assert p <= 0.01


def test_valid_seconds_flags_outages():
    n = np.ones(2000, dtype=int)
    n[500:1200] = 0  # 700 s outage
    n[1500:1600] = 0  # 100 s lull, allowed
    ok = valid_seconds(pd.DataFrame({"n_fills": n}), max_gap=300)
    assert not ok[500:1200].any() and ok[1500:1600].all() and ok[:500].all()
