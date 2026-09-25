# Data

Nothing in `data/` is committed except this file. Every input is public and the
scripts rebuild everything from it.

## Binance spot trades, 2018-04-06 to 2019-11-17 (phases 1-4, 5a)

Raw (not aggregated) trades with buyer and seller order ids, mirrored at
[Nucs/cryptocurrency-ticks-data](https://github.com/Nucs/cryptocurrency-ticks-data).
Pairs: BTCUSDT, BNBUSDT (development), ETHBTC, LTCBTC, NEOUSDT, QTUMUSDT (holdout).

```bash
git clone --filter=blob:none https://github.com/Nucs/cryptocurrency-ticks-data /data/ticks
python scripts/00_build_bars.py --mirror /data/ticks --symbols BTCUSDT BNBUSDT ETHBTC LTCBTC NEOUSDT QTUMUSDT
python scripts/00b_quote_to_usd.py --symbols ETHBTC LTCBTC
```

The loader reads blobs straight out of the partial clone, so no working tree is needed.

Known issues, handled in code:

* File names are shifted by one day relative to the UTC timestamps inside. I use the timestamps.
* A handful of files end in a truncated line; the partial row is dropped.
* Some days contain multi-hour gaps (collector outages, e.g. 2018-06-26, 2019-05-15).
  Seconds inside any gap longer than 300 s are masked, and windows with more than 20%
  masked seconds are dropped. See `data/bars/<SYM>/_manifest.csv` after the build.
* The mirror's README describes `IsBuyerMaker` loosely. I use Binance's convention
  (true means the aggressor sold), and the sign is checked empirically: signed flow and
  mid returns correlate positively on every pair.

## LOBSTER sample, AMZN 2012-06-21 (phase 5b)

Nasdaq message file plus 10-level book, 09:30 to 16:00. It is LOBSTER's public sample,
mirrored (via Git LFS) in [kpetridis24/lobsim](https://github.com/kpetridis24/lobsim)
under `sample_data/`.

## Coinbase L3 samples (phase 5b)

CoinAPI-normalized Coinbase full-channel captures for BTC-USDT (~6.5 h) and ETH-USDT
(~4.5 h), with a starting snapshot, from the same lobsim repository. The files carry
time of day but no date. The book is replayed order by order
(`kylelambda.data.l1.reconstruct_coinbase_l3`): 1,000,000 events with zero crossed
books and zero references to unknown orders.

```bash
for f in AMZN_2012-06-21_34200000_57600000_message_10.csv AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
         coinbase_btcusdt_sample_big.parquet coinbase_btcusdt_sample_big_snap.parquet \
         coinbase_ethusdt_sample_big.parquet coinbase_ethusdt_sample_big_snap.parquet; do
  curl -L -o /data/raw/$f https://media.githubusercontent.com/media/kpetridis24/lobsim/HEAD/sample_data/$f
done
python scripts/05_book_venues.py --raw /data/raw
```

## Live collection

`python -m kylelambda.collector.ws --venue coinbase --symbols BTC-USD ETH-USD --hours 24`
records raw messages to `data/ws/`. It was not run for the results in this repo; see the README.
