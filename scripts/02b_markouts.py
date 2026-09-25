"""Per-window maker markouts (inputs for phases 3 and 4).

    python scripts/02b_markouts.py --symbols BTCUSDT --windows 60 300 3600
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from kylelambda.io import DATA, bar_files, load_bars
from kylelambda.markout import window_markouts


def _day(args):
    path, windows = args
    bars = load_bars(path)
    return {W: window_markouts(bars, W).assign(date=path.stem) for W in windows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--windows", nargs="+", type=int, default=[60, 300, 3600])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for sym in a.symbols:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_day, [(p, a.windows) for p in bar_files(sym)], chunksize=4))
        for W in a.windows:
            out = DATA / "markouts" / sym / f"W{W}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([r[W] for r in res], ignore_index=True).to_parquet(out)
        print(sym, "done")


if __name__ == "__main__":
    main()
