"""Phase 4: economic tests, net of costs.

    python scripts/04_economic.py --symbols BTCUSDT BNBUSDT

(a) Quoting: gate passive liquidity on forecast vs trailing lambda.
    Grid: direction x maker fee x adverse-selection haircut kappa x hedge horizon tau.
    -> results/phase4_policies.csv, results/phase4_diffs.csv
(b) Execution: Kyle-optimal 1/lambda schedule driven by forecast vs trailing lambda.
    Grid: participation rate.
    -> results/phase4_execution.csv
All differences carry day-clustered bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from kylelambda.io import RESULTS
from kylelambda.policy import (diff_ci, execution_diff_ci, run_execution, run_policies, summarize,
                               summarize_execution)
from kylelambda.signal import load_panel

W = 300
FEES = [-0.5, 0.0, 1.0]       # bps; negative = rebate (Binance market-maker programme tiers)
KAPPAS = [1.0, 1.5]
TAUS = [10, 60, 300]
PAIRS = [("har", "trailing"), ("har", "always"), ("har", "seasonal"), ("trailing", "always")]
RATES = [0.01, 0.05, 0.10]    # base child order as a share of typical window notional
EXEC_PAIRS = [("har", "trailing"), ("har", "twap"), ("har", "seasonal"), ("trailing", "twap")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--estimator", default="kyle_usd")
    ap.add_argument("--q", type=float, default=0.8)
    a = ap.parse_args()
    lvl, dif, exe = [], [], []
    for sym in a.symbols:
        p = load_panel(sym, a.estimator, W, 1)
        for direction, fee, kappa, tau in itertools.product(["avoid_high", "avoid_low"], FEES, KAPPAS, TAUS):
            pnl = run_policies(p, W, q=a.q, tau=tau, kappa=kappa, fee_bps=fee, direction=direction)
            key = {"symbol": sym, "direction": direction, "fee_bps": fee, "kappa": kappa, "tau": tau}
            s = summarize(pnl)
            for pol, r in s.iterrows():
                lvl.append({**key, "policy": pol, **r.to_dict()})
            for x, y in PAIRS:
                for per in ["bps", "usd"]:
                    d, lo, hi, p_le0 = diff_ci(pnl, x, y, n_boot=1000, per=per)
                    dif.append({**key, "a": x, "b": y, "metric": per, "diff": d, "ci_lo": lo, "ci_hi": hi,
                                "boot_p_le0": p_le0})
        for rate in RATES:
            ex = run_execution(p, W, rate)
            s = summarize_execution(ex)
            for x, y in EXEC_PAIRS:
                d, lo, hi, p_ge0 = execution_diff_ci(ex, x, y, n_boot=1000)
                exe.append({"symbol": sym, "rate": rate, "a": x, "b": y, "cost_a_bps": s.loc[x, "cost_bps"],
                            "cost_b_bps": s.loc[y, "cost_bps"], "diff_bps": d, "ci_lo": lo, "ci_hi": hi,
                            "rel_saving": -d / s.loc[y, "cost_bps"], "boot_p_ge0": p_ge0,
                            "volume_ratio": s.loc[x, "volume_musd"] / s.loc[y, "volume_musd"]})
        print(sym, "done", flush=True)
    for path, df in [(RESULTS / "phase4_policies.csv", pd.DataFrame(lvl)), (RESULTS / "phase4_diffs.csv", pd.DataFrame(dif)),
                     (RESULTS / "phase4_execution.csv", pd.DataFrame(exe))]:
        if path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[~old.symbol.isin(a.symbols)], df], ignore_index=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
