import numpy as np
import pandas as pd

from kylelambda.bars import DAY, build_second_bars, resample
from kylelambda.data.binance import clean_trades, taker_orders
from kylelambda.markout import window_markouts


def _trades():
    # three trades: a buy at t=10 that sweeps two makers, a sell at t=20
    df = pd.DataFrame({
        "id": [1, 2, 3], "time": [10.1, 10.1, 20.5], "price": [100.0, 100.5, 99.0], "qty": [1.0, 2.0, 1.0],
        "is_buyer_maker": [False, False, True], "buyer_oid": [50, 50, 60], "seller_oid": [40, 41, 70],
        "best_match": [True] * 3,
    })
    return clean_trades(df)


def test_signs_and_taker_orders():
    tr = _trades()
    assert list(tr["sign"]) == [1, 1, -1]
    od = taker_orders(tr)
    assert len(od) == 2 and od["qty"].tolist() == [3.0, 1.0] and od["n_fills"].tolist() == [2, 1]


def test_bars_grid_and_flows():
    tr = _trades()
    b = build_second_bars(tr, taker_orders(tr), 0)
    assert len(b) == DAY
    assert b.loc[10, "sv"] == 3.0 and b.loc[20, "sv"] == -1.0
    assert np.isclose(b.loc[10, "sn"], 100 + 201)
    assert np.isclose(b.loc[10, "ssqrt_usd"], np.sqrt(301))
    assert b["mid"].notna().all()
    r = resample(b, 10)
    assert np.isclose(r["sv"].sum(), 2.0)


def test_markout_signs():
    # mid is flat at 100 then jumps to 101 at t=5; a buy of $100 at t=3 at price 100.05
    n = 20
    mid = np.full(n, 100.0); mid[5:] = 101.0
    bars = pd.DataFrame({"n_fills": np.ones(n, int), "mid": mid, "sn": 0.0, "sv": 0.0, "notional": 0.0},
                        index=pd.RangeIndex(0, n))
    bars.loc[3, ["sn", "sv", "notional"]] = [100.05, 1.0, 100.05]
    mk = window_markouts(bars, 10, taus=(1, 5))
    # after 1s mid unchanged: no adverse selection; half-spread earned 0.05/100 = 5 bps
    assert np.isclose(mk.loc[0, "as_1"], 0.0)
    assert np.isclose(mk.loc[0, "hs_1"], 1e4 * 0.05 / 100.05, rtol=1e-3)
    # after 5s mid +1%: maker who sold is down ~100 bps
    assert np.isclose(mk.loc[0, "as_5"], 100.0, rtol=1e-3)


def test_markout_half_spread_with_fx_conversion():
    # same trade as above, but the pair is quoted in BTC at 1 BTC = $20,000 and sn is in USD
    n = 20
    mid = np.full(n, 100.0)
    bars = pd.DataFrame({"n_fills": np.ones(n, int), "mid": mid, "sn": 0.0, "sv": 0.0, "notional": 0.0,
                         "fx": 20_000.0}, index=pd.RangeIndex(0, n))
    bars.loc[3, ["sn", "sv", "notional"]] = [100.05 * 20_000, 1.0, 100.05 * 20_000]
    mk = window_markouts(bars, 10, taus=(1,))
    assert np.isclose(mk.loc[0, "hs_1"], 1e4 * 0.05 / 100.05, rtol=1e-3)
