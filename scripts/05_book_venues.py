"""Phase 5b: the same audit on book-level data from other venues.

    python scripts/05_book_venues.py --raw /path/to/lobsim/sample_data

Datasets (all public samples; see data/README.md):
    AMZN     Nasdaq, LOBSTER message + 10-level book, 2012-06-21 09:30-16:00 ET
    CB_BTC   Coinbase BTC-USDT full (L3) channel, ~6.8 h from a snapshot, date not in the file
    CB_ETH   Coinbase ETH-USDT full (L3) channel, ~4.8 h

These are short. They can test whether lambda is constant within a session and how
noisy the estimator is, but they are far too short for walk-forward forecasting, so
predictability is summarized by the reliability-corrected persistence of window lambda:

    phi_true = corr(lambda_w, lambda_{w+1}) / reliability

(the lag-1 autocorrelation of the true lambda, once estimation noise is divided out).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kylelambda.audit import permutation_constancy, reliability, winsor
from kylelambda.bars import resample
from kylelambda.data.l1 import bars_from_l1, load_lobster, reconstruct_coinbase_l3
from kylelambda.estimators import FLOWS, window_lambdas
from kylelambda.io import DATA, RESULTS

EST = ["kyle_usd", "sqrt_usd", "ofi"]


def quote_ok(bars: pd.DataFrame, max_gap: int = 300) -> np.ndarray:
    q = bars["n_quotes"].to_numpy() > 0
    idx = np.flatnonzero(q)
    ok = np.ones(len(q), bool)
    edges = np.r_[-1, idx, len(q)]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a - 1 > max_gap:
            ok[a + 1:b] = False
    return ok


def load_venue(name: str, raw: Path) -> pd.DataFrame:
    if name == "AMZN":
        q, tr = load_lobster(raw / "AMZN_2012-06-21_34200000_57600000_message_10.csv",
                             raw / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv")
        return bars_from_l1(q, tr, 34200, 57600)
    pair = {"CB_BTC": "btcusdt", "CB_ETH": "ethusdt"}[name]
    snap = pd.read_parquet(raw / f"coinbase_{pair}_sample_big_snap.parquet")
    ev = pd.read_parquet(raw / f"coinbase_{pair}_sample_big.parquet")
    q, tr, st = reconstruct_coinbase_l3(snap, ev)
    print(name, st, "trades", len(tr))
    end = int(q["time"].max()) // 1800 * 1800
    return bars_from_l1(q, tr, 0, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    a = ap.parse_args()
    rows, cons, series = [], [], {}
    for name in ["AMZN", "CB_BTC", "CB_ETH"]:
        bars = load_venue(name, a.raw)
        ok = quote_ok(bars)
        out = DATA / "venues" / f"{name}_bars.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(out)
        for W in [60, 300, 900]:
            w = window_lambdas(bars, 1, W, flows=tuple(EST), ok1=ok, max_invalid=0.5)
            series[(name, W)] = w
            for est in EST:
                r = reliability(w, est, min_n=15)
                x = winsor(w[est].to_numpy(), 0.01)
                ac = pd.Series(x).autocorr(1)
                rows.append({"venue": name, "window_sec": W, "estimator": est, **r, "ac1": ac,
                             "phi_true": ac / r["reliability_sb"] if r.get("reliability_sb", 0) > 0.05 else np.nan,
                             "trades": int(bars["n_fills"].sum()), "hours": len(bars) / 3600})
        b = resample(bars, 10)
        okb = ok.reshape(-1, 10).all(1)
        y = np.where(okb, b["ret_bps"].to_numpy(), np.nan)
        y[0] = np.nan
        groups = (np.arange(len(b)) * 10) // 1800
        for est in EST:
            ratio, p = permutation_constancy(FLOWS[est](b).astype(float), y, groups, n_perm=999,
                                             rng=np.random.default_rng(1))
            cons.append({"venue": name, "estimator": est, "group_sec": 1800, "disp_ratio": ratio, "p": p})
        # does trade-based lambda move with the book? corr of window lambdas with 1/depth at the touch
        w = series[(name, 300)]
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "phase5_venues_reliability.csv", index=False)
    pd.DataFrame(cons).to_csv(RESULTS / "phase5_venues_constancy.csv", index=False)
    pd.concat({f"{k[0]}_W{k[1]}": v for k, v in series.items()}).to_parquet(DATA / "venues" / "window_lambdas.parquet")
    print(json.dumps(cons, indent=1))


if __name__ == "__main__":
    main()
