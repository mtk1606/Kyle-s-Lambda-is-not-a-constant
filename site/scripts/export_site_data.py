"""Export every number the website shows, straight from the research artifacts.

    python site/scripts/export_site_data.py        # run from the repo root

Reads results/*.csv (committed) and, for the one-day chart, data/windows + data/preds
(rebuilt by the research pipeline). Writes site/src/data/site-data.json. The page never
hard-codes a chart value; ACCURACY_AUDIT.md maps each sentence to a key in this file.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "site" / "src" / "data" / "site-data.json"
DEV = ["BTCUSDT", "BNBUSDT"]
HOLD = ["ETHBTC", "LTCBTC", "NEOUSDT", "QTUMUSDT"]
PAIRS = DEV + HOLD


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def headline_day():
    """BTCUSDT 2019-10-26: the day make_figures.py picks by rule (max realized variance in 2019, no masked windows)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kylelambda.signal import load_panel
    import os
    os.chdir(ROOT)
    W = 300
    p = load_panel("BTCUSDT", "kyle_usd", W, 1)
    w = pd.read_parquet(ROOT / "data/windows/BTCUSDT/k1_W300.parquet").drop_duplicates("t0").set_index("t0")
    daily = w.groupby("date").agg(rv=("rv_bps2", "sum"), n=("kyle_usd", "count"))
    cand = daily[(daily.index >= "2019-01-01") & (daily.n == 288)]
    date = cand["rv"].idxmax()
    t0 = int(pd.Timestamp(date).value // 10**9)
    idx = np.arange(t0, t0 + 86400, W)
    lam = w["kyle_usd"].reindex(idx)
    fc = p["f_har"].reindex(idx - W)
    const = w["kyle_usd"].loc[: t0 - 1].iloc[-7 * 288:].median()
    return {"date": date, "window_sec": W, "constant": r(const, 2),
            "minutes": [int((t - t0) // 60) for t in idx],
            "estimate": [r(v, 2) for v in lam], "forecast": [r(v, 2) for v in fc]}


def main():
    rel = pd.read_csv(R / "phase1_reliability.csv")
    cons = pd.read_csv(R / "phase1_constancy.csv")
    fc = pd.read_csv(R / "phase2_forecast.csv")
    ic = pd.read_csv(R / "phase3_ic.csv")
    dif = pd.read_csv(R / "phase4_diffs.csv")
    pol = pd.read_csv(R / "phase4_policies.csv")
    exe = pd.read_csv(R / "phase4_execution.csv")
    sc = pd.read_csv(R / "scorecard.csv")
    sc0 = pd.read_csv(R / "scorecard_as_run.csv")
    ven = pd.read_csv(R / "phase5_venues_reliability.csv")
    vcon = pd.read_csv(R / "phase5_venues_constancy.csv")

    a = rel[(rel.split == "all") & (rel.bar_sec == 1)]
    reliability = {
        est: {str(int(W)): r(a[(a.symbol == "BTCUSDT") & (a.estimator == est) & (a.window_sec == W)]["reliability_sb"].iloc[0], 3)
              for W in [60, 300, 900, 3600, 14400, 86400]}
        for est in ["kyle_usd", "sqrt_usd"]}
    btc300 = a[(a.symbol == "BTCUSDT") & (a.estimator == "kyle_usd") & (a.window_sec == 300)].iloc[0]

    pairs = {}
    for s in PAIRS:
        c = cons[cons.symbol == s]
        f1 = fc[(fc.symbol == s) & (fc.estimator == "kyle_usd") & (fc.window_sec == 300) & (fc.h == 1)].iloc[0]
        i1 = ic[(ic.symbol == s) & (ic.lead == 1)]
        get = lambda pred, tau: i1[(i1.predictor == pred) & (i1.tau == tau)].iloc[0]  # noqa: E731
        d = dif[(dif.symbol == s) & (dif.direction == "avoid_high") & (dif.tau == 60) & (dif.kappa == 1.0)
                & (dif.fee_bps == 0.0) & (dif.a == "har") & (dif.b == "trailing") & (dif.metric == "bps")].iloc[0]
        pv = pol[(pol.symbol == s) & (pol.direction == "avoid_high") & (pol.tau == 60) & (pol.kappa == 1.0)
                 & (pol.fee_bps == 0.0)].set_index("policy")["bps_per_notional"]
        ex = {}
        for b in ["trailing", "twap"]:
            e = exe[(exe.symbol == s) & (exe.rate == 0.05) & (exe.a == "har") & (exe.b == b)].iloc[0]
            ex[b] = {"diff_bps": r(e.diff_bps), "ci_lo": r(e.ci_lo), "ci_hi": r(e.ci_hi),
                     "cost_har": r(e.cost_a_bps, 3), "cost_base": r(e.cost_b_bps, 3),
                     "saving_pct": r(-100 * e.diff_bps / e.cost_b_bps, 2),
                     "saving_pct_lo": r(-100 * e.ci_hi / e.cost_b_bps, 2),
                     "saving_pct_hi": r(-100 * e.ci_lo / e.cost_b_bps, 2),
                     "significant": bool(e.ci_hi < 0)}
        pairs[s] = {
            "set": "development" if s in DEV else "holdout",
            "reliability_5m": r(a[(a.symbol == s) & (a.estimator == "kyle_usd") & (a.window_sec == 300)]["reliability_sb"].iloc[0], 3),
            "constancy_reject_1pct": r((c.kyle_usd_p <= 0.01).mean(), 3),
            "constancy_reject_1pct_sqrt": r((c.sqrt_usd_p <= 0.01).mean(), 3),
            "days": int(len(c)),
            "r2": {k: r(f1[f"r2_{k}"], 4) for k in ["last", "trailing_cal", "seasonal_cal", "har"]},
            "dm_vs_trailing": r(f1.dm_har_vs_trailing_cal, 1), "dm_vs_seasonal": r(f1.dm_har_vs_seasonal_cal, 1),
            "ic_forecast_lambda": r(get("f_har->lambda", -1).ic_daily, 3),
            "ic_trailing_lambda": r(get("f_trailing->lambda", -1).ic_daily, 3),
            "ic_forecast_as": r(get("f_har", 60).ic_daily, 3), "ic_forecast_as_t": r(get("f_har", 60).ic_daily_t, 1),
            "ic_vol_as": r(get("f_rv", 60).ic_daily, 3),
            "quote_diff_bps": r(d["diff"]), "quote_ci": [r(d.ci_lo), r(d.ci_hi)],
            "quote_bps": {k: r(pv[k], 3) for k in ["always", "trailing", "har", "oracle"]},
            "execution": ex,
        }

    def card(df):
        out = {}
        for _, row in df.iterrows():
            out[row.symbol] = {f"P{k}": bool(row[f"P{k}"]) for k in range(1, 8)}
            out[row.symbol]["P6_diff_bps"] = r(row.P6_diff_bps)
            out[row.symbol]["P7_diff_bps"] = r(row.P7_diff_bps)
            out[row.symbol]["P6_ci"] = row.P6_ci
            out[row.symbol]["P7_ci"] = row.P7_ci
        return out

    horizons = {}
    for s in DEV:
        d = fc[(fc.symbol == s) & (fc.estimator == "kyle_usd") & (fc.window_sec == 300)].sort_values("h")
        horizons[s] = {"minutes": [int(x) for x in d.horizon_sec / 60],
                       **{k: [r(v, 4) for v in d[f"r2_{k}"]] for k in ["har", "trailing_cal", "seasonal_cal"]},
                       "ceiling": r(d.reliability_sb.iloc[0], 3)}

    venues = {}
    for v in ["AMZN", "CB_BTC", "CB_ETH"]:
        x = ven[(ven.venue == v) & (ven.window_sec == 300)].set_index("estimator")
        cc = vcon[vcon.venue == v].set_index("estimator")
        venues[v] = {"reliability_5m": {e: r(x.loc[e, "reliability_sb"], 4) for e in ["kyle_usd", "sqrt_usd", "ofi"]},
                     "trades": int(x["trades"].iloc[0]), "hours": r(x["hours"].iloc[0], 1),
                     "constancy_p": {e: r(cc.loc[e, "p"], 3) for e in ["kyle_usd", "sqrt_usd", "ofi"]},
                     "dispersion": r(cc.loc["kyle_usd", "disp_ratio"], 2)}

    trades = 0
    for s in PAIRS:
        man = ROOT / "data" / "bars" / s / "_manifest.csv"
        trades += int(pd.read_csv(man)["n_trades"].sum()) if man.exists() else 0
    commits = []
    log = subprocess.run(["git", "log", "--reverse", "--format=%h|%aI|%s"], cwd=ROOT, capture_output=True, text=True).stdout
    for line in log.strip().splitlines():
        h, t, msg = line.split("|", 2)
        commits.append({"hash": h, "time": t, "subject": msg})
    tests = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    n_tests = sum(1 for ln in tests.splitlines() if "::" in ln)

    verdict = {}
    for k in range(1, 8):
        h = sc[sc.set == "holdout"][f"P{k}"].sum()
        h0 = sc0[sc0.set == "holdout"][f"P{k}"].sum()
        verdict[f"P{k}"] = {"holdout_pass": int(h), "as_run_pass": int(h0), "holds": bool(h >= 3), "as_run_holds": bool(h0 >= 3)}

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kylelambda.kyle import kyle_equilibrium
    kyle = {str(N): [r(v, 4) for v in kyle_equilibrium(N).lam] for N in (4, 20, 200)}

    data = {
        "kyle_equilibrium": kyle,
        "meta": {"trades": trades, "pairs": PAIRS, "development": DEV, "holdout": HOLD,
                 "period": ["2018-04-06", "2019-11-17"], "tests": n_tests, "commits": commits},
        "btc_5m": {"reliability": r(btc300.reliability_sb, 3), "noise_share": r(1 - btc300.reliability_sb, 3),
                   "rel_noise": r(btc300.rel_noise, 2), "frac_negative": r(btc300.frac_negative, 3),
                   "median_lambda": r(btc300.median_lambda, 1)},
        "reliability_btc": reliability, "horizons": horizons, "pairs": pairs,
        "scorecard": card(sc), "scorecard_as_run": card(sc0), "verdict": verdict,
        "venues": venues, "headline_day": headline_day(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print("wrote", OUT, f"{OUT.stat().st_size / 1024:.0f} KB; trades={trades:,} tests={n_tests}")


if __name__ == "__main__":
    main()
