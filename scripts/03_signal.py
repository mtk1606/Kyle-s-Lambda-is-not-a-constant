"""Phase 3: does forecast lambda predict adverse selection on passive fills?

    python scripts/03_signal.py --symbols BTCUSDT BNBUSDT

Writes results/phase3_ic.csv (IC by predictor, markout horizon, forecast lead),
results/phase3_incremental.csv (does the forecast add over trailing lambda and volatility).
"""
from __future__ import annotations

import argparse

import pandas as pd

from kylelambda.io import RESULTS
from kylelambda.signal import ic_summary, incremental_regression, load_panel

W = 300
LEADS = [1, 2, 3, 6, 12, 24]
TAUS = [10, 60, 300]
PREDICTORS = ["f_har", "f_trailing", "f_seasonal", "f_last", "f_rv", "f_abs_oi", "f_amihud"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--estimator", default="kyle_usd")
    a = ap.parse_args()
    ic_rows, inc_rows = [], []
    for sym in a.symbols:
        for h in LEADS:
            p = load_panel(sym, a.estimator, W, h)
            for tau in TAUS:
                for f in PREDICTORS:
                    ic_rows.append({"symbol": sym, "estimator": a.estimator, "window_sec": W, "lead": h,
                                    "lead_sec": h * W, "tau": tau, "predictor": f,
                                    **ic_summary(p, f, f"as_{tau}")})
                if h == 1:
                    # does the forecast carry information beyond what a desk already watches?
                    cols = ["f_har", "f_trailing", "f_rv", "f_abs_oi"]
                    r = incremental_regression(p, f"as_{tau}", cols, lags=12)
                    for c, v in r.iterrows():
                        inc_rows.append({"symbol": sym, "tau": tau, "term": c, **v.to_dict()})
            # the forecast of lambda should also predict realized lambda: sanity check
            for f in ["f_har", "f_trailing", "f_seasonal"]:
                ic_rows.append({"symbol": sym, "estimator": a.estimator, "window_sec": W, "lead": h,
                                "lead_sec": h * W, "tau": -1, "predictor": f"{f}->lambda",
                                **ic_summary(p, f, "lambda_next")})
            print(sym, h, "done", flush=True)
    RESULTS.mkdir(exist_ok=True)
    ic = pd.DataFrame(ic_rows)
    inc = pd.DataFrame(inc_rows)
    for path, df in [(RESULTS / "phase3_ic.csv", ic), (RESULTS / "phase3_incremental.csv", inc)]:
        if path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[~old.symbol.isin(a.symbols)], df], ignore_index=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
