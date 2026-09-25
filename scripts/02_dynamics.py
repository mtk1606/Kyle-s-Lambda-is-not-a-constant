"""Phase 2: is lambda predictable, and is it more than intraday seasonality?

    python scripts/02_dynamics.py --symbols BTCUSDT BNBUSDT

Writes
    data/preds/<SYM>/<est>_W<window>_h<h>.parquet   walk-forward forecasts (reused by phases 3-4)
    results/phase2_forecast.csv                      OOS R^2 / DM tests per config
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from kylelambda.forecast import oos_r2, to_grid, walk_forward
from kylelambda.io import DATA, RESULTS, load_windows
from kylelambda.stats import diebold_mariano

log = logging.getLogger("phase2")
# window length -> horizons (in windows). 1-minute windows cover seconds-to-minutes,
# 5-minute windows minutes-to-hours, 1-hour windows hours-to-a-day.
GRID = {60: [1, 5, 15, 60], 300: [1, 2, 3, 6, 12, 24], 3600: [1, 2, 4, 8, 24]}
ESTS = ["kyle_usd", "sqrt_usd"]


def _run(args):
    sym, est, W, h = args
    g = to_grid(load_windows(sym, 1, W), W)
    p = walk_forward(g, est, W, h)
    out = DATA / "preds" / sym / f"{est}_W{W}_h{h}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out)
    p = p.dropna(subset=["target", "trailing", "seasonal", "trailing_cal", "seasonal_cal", "har", "last"])
    y = p["target"].to_numpy()
    c = p["constant"].to_numpy()
    row = {"symbol": sym, "estimator": est, "window_sec": W, "h": h, "horizon_sec": W * h, "n": len(p)}
    for m in ["last", "trailing", "seasonal", "trailing_cal", "seasonal_cal", "har"]:
        row[f"r2_{m}"] = oos_r2(y, p[m].to_numpy(), c)
    row["r2_har_over_seasonal"] = oos_r2(y, p["har"].to_numpy(), p["seasonal_cal"].to_numpy())
    row["r2_har_over_trailing"] = oos_r2(y, p["har"].to_numpy(), p["trailing_cal"].to_numpy())
    for m in ["trailing_cal", "seasonal_cal", "constant"]:
        t, pv = diebold_mariano(y - p["har"].to_numpy(), y - p[m].to_numpy(), h)
        row[f"dm_har_vs_{m}"] = t
        row[f"dm_p_har_vs_{m}"] = pv
    row["trailing_halflife_mode"] = float(p["hl"].mode().iloc[0])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--core", action="store_true",
                    help="only the configs phases 3-4 need (5-minute windows, kyle_usd) plus 1m/1h at h=1")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.core:
        jobs = [(s, "kyle_usd", 300, h) for s in a.symbols for h in GRID[300]]
        jobs += [(s, "kyle_usd", W, 1) for s in a.symbols for W in (60, 3600)]
    else:
        jobs = [(s, e, W, h) for s in a.symbols for e in ESTS for W, hs in GRID.items() for h in hs]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_run, jobs))
    df = pd.DataFrame(rows)
    rel = pd.read_csv(RESULTS / "phase1_reliability.csv")
    rel = rel[(rel.split == "all") & (rel.bar_sec == 1)][["symbol", "estimator", "window_sec", "reliability_sb"]]
    df = df.merge(rel, on=["symbol", "estimator", "window_sec"], how="left")
    df["har_share_of_ceiling"] = df["r2_har"] / df["reliability_sb"]
    path = RESULTS / "phase2_forecast.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.symbol.isin(a.symbols)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    log.info("\n%s", df.round(3).to_string())


if __name__ == "__main__":
    main()
