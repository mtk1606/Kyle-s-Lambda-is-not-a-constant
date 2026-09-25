"""Score the pre-registered predictions (docs/PREREGISTRATION.md) on every pair.

    python scripts/06_scorecard.py
"""
from __future__ import annotations

import pandas as pd

from kylelambda.io import RESULTS

DEV = ["BTCUSDT", "BNBUSDT"]
HOLDOUT = ["ETHBTC", "LTCBTC", "NEOUSDT", "QTUMUSDT"]


def main():
    cons = pd.read_csv(RESULTS / "phase1_constancy.csv")
    rel = pd.read_csv(RESULTS / "phase1_reliability.csv")
    fc = pd.read_csv(RESULTS / "phase2_forecast.csv")
    ic = pd.read_csv(RESULTS / "phase3_ic.csv")
    dif = pd.read_csv(RESULTS / "phase4_diffs.csv")
    exe = pd.read_csv(RESULTS / "phase4_execution.csv")
    rows = []
    for sym in DEV + HOLDOUT:
        r = {"symbol": sym, "set": "dev" if sym in DEV else "holdout"}
        c = cons[cons.symbol == sym]
        r["P1_reject_rate"] = (c["kyle_usd_p"] <= 0.01).mean()
        r["P1"] = r["P1_reject_rate"] >= 0.10
        x = rel[(rel.symbol == sym) & (rel.estimator == "kyle_usd") & (rel.bar_sec == 1) & (rel.window_sec == 300) & (rel.split == "all")]
        r["P2_reliability"] = x["reliability_sb"].iloc[0]
        r["P2"] = r["P2_reliability"] < 0.6
        f = fc[(fc.symbol == sym) & (fc.estimator == "kyle_usd") & (fc.window_sec == 300) & (fc.h == 1)].iloc[0]
        r["P3_dm_trailing"], r["P3_dm_seasonal"] = f["dm_har_vs_trailing_cal"], f["dm_har_vs_seasonal_cal"]
        r["P3"] = (f["dm_har_vs_trailing_cal"] > 2) and (f["dm_har_vs_seasonal_cal"] > 2)
        i = ic[(ic.symbol == sym) & (ic.lead == 1)]
        a = i[(i.predictor == "f_har") & (i.tau == 60)].iloc[0]
        r["P4_ic"], r["P4_t"] = a["ic_daily"], a["ic_daily_t"]
        r["P4"] = (a["ic_daily"] > 0) and (a["ic_daily_t"] > 2)
        lam_har = i[i.predictor == "f_har->lambda"]["ic_daily"].iloc[0]
        r["P5_ic_har"] = lam_har
        tr = i[i.predictor == "f_trailing->lambda"]["ic_daily"]
        r["P5_ic_trailing"] = tr.iloc[0] if len(tr) else float("nan")
        r["P5"] = lam_har > r["P5_ic_trailing"]
        d = dif[(dif.symbol == sym) & (dif.direction == "avoid_high") & (dif.tau == 60) & (dif.kappa == 1.0)
                & (dif.fee_bps == 0.0) & (dif.a == "har") & (dif.b == "trailing") & (dif.metric == "bps")].iloc[0]
        r["P6_diff_bps"], r["P6_ci"] = d["diff"], f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
        r["P6"] = d["ci_lo"] > 0
        e = exe[(exe.symbol == sym) & (exe.rate == 0.05) & (exe.a == "har") & (exe.b == "trailing")].iloc[0]
        r["P7_diff_bps"], r["P7_ci"], r["P7_rel_saving"] = e["diff_bps"], f"[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]", e["rel_saving"]
        r["P7"] = e["ci_hi"] < 0
        rows.append(r)
    sc = pd.DataFrame(rows)
    sc.to_csv(RESULTS / "scorecard.csv", index=False)
    h = sc[sc.set == "holdout"]
    verdict = {p: f"{int(h[p].sum())}/4 -> {'HOLDS' if h[p].sum() >= 3 else 'FAILS'}" for p in [f"P{k}" for k in range(1, 8)]}
    pd.Series(verdict).to_csv(RESULTS / "scorecard_verdict.csv", header=["holdout"])
    print(sc.round(3).to_string())
    print(verdict)


if __name__ == "__main__":
    main()
