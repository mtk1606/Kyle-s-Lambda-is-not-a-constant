# Pre-registration: holdout pairs

I'm writing this after running every phase on the two development pairs (BTCUSDT,
BNBUSDT) and before computing anything past the bar build on the four holdout pairs.
The commit that adds this file freezes the code; the holdout run uses exactly that
commit with no parameter changes. Git history is the timestamp.

**Why this exists.** The development results did not come out the way my original
thesis said they would (see "What changed" below). Once I had seen those results, any
reframing I chose was fitted to them. The only honest way to find out whether the
reframed claims are real is to fix them in writing and test them on data I haven't
looked at.

## Holdout set

Binance spot, same period (2018-04-06 to 2019-11-17), same pipeline:
**ETHBTC, LTCBTC, NEOUSDT, QTUMUSDT**. BTC-quoted pairs are converted to USD with the
BTCUSDT mid of the same second (`scripts/00b_quote_to_usd.py`), so lambda is in bps per $M
everywhere.

Commands, in order:

```bash
python scripts/01_audit.py     --symbols ETHBTC LTCBTC NEOUSDT QTUMUSDT
python scripts/02b_markouts.py --symbols ETHBTC LTCBTC NEOUSDT QTUMUSDT --windows 300
python scripts/02_dynamics.py  --symbols ETHBTC LTCBTC NEOUSDT QTUMUSDT --core
python scripts/03_signal.py    --symbols ETHBTC LTCBTC NEOUSDT QTUMUSDT
python scripts/04_economic.py  --symbols ETHBTC LTCBTC NEOUSDT QTUMUSDT
python scripts/06_scorecard.py
```

## Predictions

Each one is scored per pair. "Holds" means it holds on at least 3 of the 4 pairs.

| # | Phase | Prediction | Test |
|---|-------|------------|------|
| P1 | 1 | Constant intraday lambda is rejected far more often than chance | `kyle_usd` permutation test (10 s bars, hourly groups) rejects at 1% on at least 10% of days |
| P2 | 1 | A 5-minute rolling lambda is mostly noise | split-half reliability of `kyle_usd`, 1 s bars, 5 min windows, is below 0.6 |
| P3 | 2 | Lambda is predictable beyond seasonality and beyond a tuned trailing estimate | 5 min windows, h = 1: HAR beats `trailing_cal` and `seasonal_cal`, Diebold-Mariano t > 2 for both |
| P4 | 3 | *Original thesis, expected to fail:* forecast lambda predicts maker adverse selection | daily IC of `f_har` with `as_60` > 0 with NW t > 2 |
| P5 | 3 | Forecast lambda predicts realized lambda better than trailing lambda does | daily IC(`f_har`, realized lambda) > daily IC(`f_trailing`, realized lambda) at lead 1 |
| P6 | 4 | *Original thesis, expected to fail:* a quoting rule that steps away from high forecast lambda beats the same rule on trailing lambda | `avoid_high`, tau = 60 s, kappa = 1: HAR minus trailing net bps > 0 with the 95% CI excluding 0 |
| P7 | 4 | A Kyle-optimal execution schedule driven by forecast lambda is cheaper than the same schedule driven by trailing lambda | rate = 5%: cost(HAR) minus cost(trailing) < 0 with the 95% CI excluding 0 |

P4 and P6 are the claims as I first wrote them. I expect them to fail: on the
development pairs, P4 went negative on BTCUSDT and positive on BNBUSDT, and P6's CI
included zero on both. I'm keeping them in the scorecard anyway, because dropping
failed predictions is how a research log turns into marketing.

## What changed between the original thesis and this document

Original: *forecast lambda predicts adverse selection, so a quoting rule should
condition on it.*

What the development pairs showed:

1. Lambda is not constant, the rolling estimate is mostly noise at the horizons people
   use it, and it is predictable beyond seasonality. This part held.
2. The forecast predicts **realized lambda** well (daily IC 0.15 BTCUSDT, 0.26 BNBUSDT,
   versus 0.05 and 0.04 for trailing lambda). It does **not** reliably predict maker
   adverse selection per dollar. Trailing realized volatility does (IC 0.10 and 0.12).
3. Even an oracle that knows next window's realized lambda does not help a maker who
   avoids high-lambda windows (BTCUSDT: -0.84 bps vs -0.74 bps for always quoting).
   High-lambda windows are thin, and thin windows are not where passive fills get picked
   off per dollar. Makers seem to already price lambda into the spread.
4. Lambda is by definition the price of impact, so the natural place for the forecast to
   earn money is on the other side: sizing child orders for someone who has to trade.
   That is P7.
