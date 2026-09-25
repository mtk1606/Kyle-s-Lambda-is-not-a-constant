"""Build 1-second bars for every Binance day in the mirror.

    python scripts/00_build_bars.py --mirror /path/to/cryptocurrency-ticks-data --symbols BTCUSDT ETHBTC

Output: data/bars/<SYMBOL>/<YYYY-MM-DD>.parquet (UTC date of the first trade), plus
data/bars/<SYMBOL>/_manifest.csv with trade counts and data-quality flags per day.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from kylelambda.bars import DAY, build_second_bars
from kylelambda.data.binance import BinanceMirror, taker_orders

log = logging.getLogger("build_bars")
OUT = Path("data/bars")


def _one(args):
    mirror, rel, sym = args
    try:
        tr = BinanceMirror(Path(mirror)).load_day(rel)
    except Exception as e:  # corrupt zip or missing blob: record and move on
        return {"file": rel, "status": f"error: {e!r}"}
    if tr.empty:
        return {"file": rel, "status": "empty"}
    day_start = int(tr["time"].iloc[len(tr) // 2] // DAY * DAY)
    date = pd.Timestamp(day_start, unit="s").strftime("%Y-%m-%d")
    od = taker_orders(tr)
    bars = build_second_bars(tr, od, day_start)
    out = OUT / sym / f"{date}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out, compression="zstd")
    in_day = ((tr["time"] >= day_start) & (tr["time"] < day_start + DAY)).mean()
    active = (bars["n_fills"] > 0).mean()
    gaps = np.diff(np.r_[0, np.flatnonzero(bars["n_fills"].to_numpy()), DAY - 1]).max()
    return {"file": rel, "date": date, "status": "ok", "n_trades": len(tr), "n_orders": len(od),
            "frac_in_day": round(float(in_day), 4), "frac_active_sec": round(float(active), 4),
            "max_gap_sec": int(gaps), "px_median": float(tr["price"].median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for sym in a.symbols:
        files = BinanceMirror(Path(a.mirror)).files(sym)
        log.info("%s: %d files", sym, len(files))
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(_one, [(a.mirror, f, sym) for f in files], chunksize=4))
        man = pd.DataFrame(rows)
        man.to_csv(OUT / sym / "_manifest.csv", index=False)
        bad = man[man["status"] != "ok"]
        log.info("%s: %d ok, %d problems", sym, (man["status"] == "ok").sum(), len(bad))


if __name__ == "__main__":
    main()
