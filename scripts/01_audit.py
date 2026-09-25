"""Phase 1: replicate the standard lambda estimators and audit their noise.

    python scripts/01_audit.py --symbols BTCUSDT

Writes
    data/windows/<SYM>/k{bar}_W{window}.parquet   per-window estimates, reused by later phases
    results/phase1_reliability.csv                 reliability grid (estimator x bar x window x regime)
    results/phase1_constancy.csv                   per-day permutation test of a constant intraday lambda
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from kylelambda.audit import permutation_constancy, reliability
from kylelambda.bars import resample
from kylelambda.estimators import FLOWS, valid_seconds, window_lambdas
from kylelambda.io import RESULTS, bar_files, load_bars, window_path

log = logging.getLogger("phase1")
BARS = [1, 5, 15, 60]
WINDOWS = [60, 300, 900, 3600, 14400, 86400]
ESTIMATORS = ["kyle_usd", "sqrt_usd", "count"]


def _day(path):
    bars = load_bars(path)
    out = {}
    for k in BARS:
        for W in WINDOWS:
            if W // k < 5:
                continue
            df = window_lambdas(bars, k, W, flows=tuple(ESTIMATORS))
            df["date"] = path.stem
            out[(k, W)] = df
    # constancy: 10s bars, one group per UTC hour, bars inside outages dropped
    ok = valid_seconds(bars).reshape(-1, 10).all(1)
    b = resample(bars, 10)
    hours = (np.arange(len(b)) * 10) // 3600
    y = np.where(ok, b["ret_bps"].to_numpy(), np.nan)
    y[0] = np.nan
    rng = np.random.default_rng(abs(hash(path.stem)) % 2**32)
    cons = {"date": path.stem}
    for f in ["kyle_usd", "sqrt_usd"]:
        ratio, p = permutation_constancy(FLOWS[f](b).astype(float), y, hours, n_perm=199, rng=rng)
        cons[f + "_disp_ratio"] = ratio
        cons[f + "_p"] = p
    return out, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rel_rows, cons_rows = [], []
    for sym in a.symbols:
        files = bar_files(sym)
        log.info("%s: %d days", sym, len(files))
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_day, files, chunksize=2))
        cons = pd.DataFrame([c for _, c in res])
        cons["symbol"] = sym
        cons_rows.append(cons)
        # daily realized-vol terciles define the regime split
        for (k, W) in res[0][0]:
            df = pd.concat([r[0][(k, W)] for r in res], ignore_index=True)
            path = window_path(sym, k, W)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            day_rv = df.groupby("date")["rv_bps2"].sum()
            terc = pd.qcut(day_rv, 3, labels=["low_vol", "mid_vol", "high_vol"])
            df["vol_regime"] = df["date"].map(terc).astype(str)
            hour = (df["t0"] % 86400) // 3600
            df["session"] = np.select([hour < 8, hour < 13], ["asia", "europe"], "us")
            df["period"] = np.where(df["date"] < "2019-01-01", "2018", "2019")
            for est in ESTIMATORS:
                groups = [("all", "all", df)]
                if W < 86400:
                    groups += [("vol_regime", g, d) for g, d in df.groupby("vol_regime")]
                    groups += [("session", g, d) for g, d in df.groupby("session")]
                groups += [("period", g, d) for g, d in df.groupby("period")]
                for gname, gval, d in groups:
                    rel_rows.append({"symbol": sym, "estimator": est, "bar_sec": k, "window_sec": W,
                                     "split": gname, "group": gval, **reliability(d, est)})
            log.info("%s k=%d W=%d done", sym, k, W)
    RESULTS.mkdir(exist_ok=True)
    for path, df in [(RESULTS / "phase1_reliability.csv", pd.DataFrame(rel_rows)),
                     (RESULTS / "phase1_constancy.csv", pd.concat(cons_rows))]:
        if path.exists():  # keep other symbols' rows
            old = pd.read_csv(path)
            df = pd.concat([old[~old.symbol.isin(a.symbols)], df], ignore_index=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
