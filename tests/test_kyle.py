import numpy as np

from kylelambda.kyle import kyle_equilibrium


def test_single_auction_closed_form():
    # Kyle (1985) Theorem 1: lambda = sigma_v / (2 sigma_u), beta = sigma_u / sigma_v, Sigma_1 = Sigma_0 / 2
    e = kyle_equilibrium(1, sigma_v=2.0, sigma_u=0.5)
    assert np.isclose(e.lam[0], 2.0 / (2 * 0.5))
    assert np.isclose(e.beta[0], 0.5 / 2.0)
    assert np.isclose(e.sigma[1], 4.0 / 2)


def test_continuous_limit_constant_depth():
    # With many auctions lambda is flat at sigma_v / sigma_u away from the close,
    # and (almost) all information is in the price by T.
    e = kyle_equilibrium(400, sigma_v=1.0, sigma_u=1.0)
    interior = e.lam[: int(0.9 * len(e.lam))]
    assert np.allclose(interior, 1.0, atol=0.02)
    assert interior.max() / interior.min() < 1.01
    assert e.sigma[-1] < 0.01


def test_second_order_condition_holds():
    e = kyle_equilibrium(50)
    assert np.all(e.lam > 0) and np.all(np.diff(e.sigma) < 0)
