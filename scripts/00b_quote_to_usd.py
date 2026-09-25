"""Convert bars of BTC-quoted pairs (ETHBTC, LTCBTC) to USD using the BTCUSDT mid of the same second.

    python scripts/00b_quote_to_usd.py --symbols ETHBTC LTCBTC

Only the dollar-denominated flow columns change (notional, sn, ssqrt_usd); returns are
in bps of the pair's own mid and are left alone, so lambda comes out in bps per $M
and is comparable across pairs. The conversion rate is kept as column `fx`.
"""
import argparse

import numpy as np
import pandas as pd

from kylelambda.io import DATA, bar_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    a = ap.parse_args()
    for sym in a.symbols:
        n = 0
        for p in bar_files(sym):
            ref = DATA / "bars" / "BTCUSDT" / p.name
            if not ref.exists():
                p.unlink()  # no USD reference for that day: drop it rather than mix units
                continue
            b = pd.read_parquet(p)
            px = pd.read_parquet(ref, columns=["mid"])["mid"].reindex(b.index).ffill().bfill().to_numpy()
            if "usd_converted" in b.columns:
                if "fx" not in b.columns:  # files converted before the fx column existed
                    b["fx"] = px
                    b.to_parquet(p, compression="zstd")
                continue
            b["notional"] *= px
            b["sn"] *= px
            b["ssqrt_usd"] *= np.sqrt(px)
            # USD per unit of the quote currency; markout.py needs it to put
            # sn (now USD) and mid * sv (still in the quote currency) in the same units
            b["fx"] = px
            b["usd_converted"] = True
            b.to_parquet(p, compression="zstd")
            n += 1
        print(sym, "converted", n, "days")


if __name__ == "__main__":
    main()
