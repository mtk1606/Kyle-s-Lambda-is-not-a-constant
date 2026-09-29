"""Check every numeric claim on the page against the research artifacts, and write ACCURACY_AUDIT.md.

    python site/scripts/verify_claims.py        # from the repo root; exits 1 on any mismatch

Each claim has: the exact phrase that must appear in site/index.html, the value recomputed
from results/*.csv (or git, or pytest), and the file it comes from. Qualitative claims
(what was and was not done) are listed with the evidence a reader can check by hand.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
PAGE = (ROOT / "site" / "index.html").read_text()
TEXT = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", PAGE)))
DATA = json.loads((ROOT / "site" / "src" / "data" / "site-data.json").read_text())

rel = pd.read_csv(R / "phase1_reliability.csv")
cons = pd.read_csv(R / "phase1_constancy.csv")
fc = pd.read_csv(R / "phase2_forecast.csv")
ic = pd.read_csv(R / "phase3_ic.csv")
pol = pd.read_csv(R / "phase4_policies.csv")
exe = pd.read_csv(R / "phase4_execution.csv")
sc = pd.read_csv(R / "scorecard.csv")
sc0 = pd.read_csv(R / "scorecard_as_run.csv")
ven = pd.read_csv(R / "phase5_venues_reliability.csv")
vcon = pd.read_csv(R / "phase5_venues_constancy.csv")
HOLD = ["ETHBTC", "LTCBTC", "NEOUSDT", "QTUMUSDT"]
ALL = ["BTCUSDT", "BNBUSDT"] + HOLD


def rel_of(sym, W, est="kyle_usd"):
    return rel[(rel.symbol == sym) & (rel.estimator == est) & (rel.bar_sec == 1) & (rel.window_sec == W) & (rel.split == "all")]["reliability_sb"].iloc[0]


def f1(sym, col):
    return fc[(fc.symbol == sym) & (fc.estimator == "kyle_usd") & (fc.window_sec == 300) & (fc.h == 1)][col].iloc[0]


def icv(sym, pred, tau):
    return ic[(ic.symbol == sym) & (ic.lead == 1) & (ic.predictor == pred) & (ic.tau == tau)]["ic_daily"].iloc[0]


def rej(sym, col="kyle_usd_p"):
    return (cons[cons.symbol == sym][col] <= 0.01).mean()


def ex(sym, b="trailing"):
    return exe[(exe.symbol == sym) & (exe.rate == 0.05) & (exe.a == "har") & (exe.b == b)].iloc[0]


def qpol(sym):
    return pol[(pol.symbol == sym) & (pol.direction == "avoid_high") & (pol.tau == 60) & (pol.kappa == 1.0) & (pol.fee_bps == 0.0)].set_index("policy")["bps_per_notional"]


trades = sum(int(pd.read_csv(ROOT / "data" / "bars" / s / "_manifest.csv")["n_trades"].sum()) for s in ALL)
n_tests = sum(1 for ln in subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.splitlines() if "::" in ln)
commits = {ln.split("|")[0]: ln.split("|")[1] for ln in subprocess.run(
    ["git", "log", "--format=%h|%aI"], cwd=ROOT, capture_output=True, text=True).stdout.split()}
holdout_pass = {k: int(sc[sc.set == "holdout"][k].sum()) for k in [f"P{i}" for i in range(1, 8)]}
asrun_pass = {k: int(sc0[sc0.set == "holdout"][k].sum()) for k in [f"P{i}" for i in range(1, 8)]}
oracle_worse = sum(qpol(s)["oracle"] < qpol(s)["always"] for s in ALL)
sig = [s for s in ALL if ex(s).ci_hi < 0]
pct_save = {s: -100 * ex(s).diff_bps / ex(s).cost_b_bps for s in ALL}
lam_ic_hold = [icv(s, "f_har->lambda", -1) for s in HOLD]
lam_ic_all = [icv(s, "f_har->lambda", -1) for s in ALL]
tr_ic_all = [icv(s, "f_trailing->lambda", -1) for s in ALL]
vol_ic = [icv(s, "f_rv", 60) for s in ALL]
vol_t = [ic[(ic.symbol == s) & (ic.lead == 1) & (ic.predictor == "f_rv") & (ic.tau == 60)]["ic_daily_t"].iloc[0] for s in ALL]

dev_dm = fc[fc.symbol.isin(["BTCUSDT", "BNBUSDT"])]["dm_har_vs_seasonal_cal"]

# (phrase on the page, condition that must hold, source, what was recomputed)
CLAIMS = [
    ("351M", round(trades / 1e6) == 351, "data/bars/*/_manifest.csv", f"{trades:,} trades"),
    ("6 Binance spot pairs", len(ALL) == 6, "scripts/00_build_bars.py manifests", ", ".join(ALL)),
    ("2018-04-06 to 2019-11-17", DATA["meta"]["period"] == ["2018-04-06", "2019-11-17"], "data/README.md, manifests", "first and last bar dates"),
    ("about 57% noise", round(100 * (1 - rel_of("BTCUSDT", 300))) == 57, "results/phase1_reliability.csv", f"1 - {rel_of('BTCUSDT', 300):.3f}"),
    ("0.43", round(rel_of("BTCUSDT", 300), 2) == 0.43, "results/phase1_reliability.csv", "BTCUSDT kyle 1s/5m reliability"),
    ("0.69", round(rel_of("BTCUSDT", 3600), 2) == 0.69, "results/phase1_reliability.csv", "BTCUSDT kyle 1s/1h reliability"),
    ("5.4%", round(100 * rel[(rel.symbol == "BTCUSDT") & (rel.estimator == "kyle_usd") & (rel.bar_sec == 1) & (rel.window_sec == 300) & (rel.split == "all")]["frac_negative"].iloc[0], 1) == 5.4, "results/phase1_reliability.csv", "share of negative 5-min estimates"),
    ("27% to 53% of days rejected", round(100 * min(rej("BTCUSDT"), rej("BNBUSDT"), rej("BTCUSDT", "sqrt_usd_p"), rej("BNBUSDT", "sqrt_usd_p"))) == 27
     and round(100 * max(rej("BTCUSDT"), rej("BNBUSDT"), rej("BTCUSDT", "sqrt_usd_p"), rej("BNBUSDT", "sqrt_usd_p"))) == 53, "results/phase1_constancy.csv", "dev pairs, kyle and sqrt estimators"),
    ("7% to 31% did", round(100 * min(rej(s) for s in HOLD)) == 7 and round(100 * max(rej(s) for s in HOLD)) == 31, "results/phase1_constancy.csv", "holdout pairs, kyle estimator"),
    ("p = 0.001", vcon[(vcon.venue == "AMZN") & (vcon.estimator == "kyle_usd")]["p"].iloc[0] <= 0.001, "results/phase5_venues_constancy.csv", "AMZN permutation p"),
    ("9.9%", round(100 * f1("BTCUSDT", "r2_har"), 1) == 9.9, "results/phase2_forecast.csv", "BTCUSDT W=300 h=1 HAR R²"),
    ("7.9%", round(100 * f1("BTCUSDT", "r2_trailing_cal"), 1) == 7.9, "results/phase2_forecast.csv", "calibrated trailing R²"),
    ("5.1%", round(100 * f1("BTCUSDT", "r2_seasonal_cal"), 1) == 5.1, "results/phase2_forecast.csv", "calibrated seasonal R²"),
    ("t-statistics are about 20", 15 < f1("BTCUSDT", "dm_har_vs_trailing_cal") < 25 and 15 < f1("BTCUSDT", "dm_har_vs_seasonal_cal") < 25, "results/phase2_forecast.csv", f"DM {f1('BTCUSDT', 'dm_har_vs_trailing_cal'):.1f} / {f1('BTCUSDT', 'dm_har_vs_seasonal_cal'):.1f}"),
    ("R² = −64%", round(100 * f1("BTCUSDT", "r2_last")) == -64, "results/phase2_forecast.csv", "last-window forecast R²"),
    ("ceiling for any forecast is about 43%", round(100 * rel_of("BTCUSDT", 300)) == 43, "results/phase1_reliability.csv", "noise ceiling = reliability"),
    ("IC 0.10 to 0.17 on the holdout", round(min(lam_ic_hold), 2) == 0.10 and round(max(lam_ic_hold), 2) == 0.17, "results/phase3_ic.csv", ", ".join(f"{v:.3f}" for v in lam_ic_hold)),
    ("at most 0.05", max(tr_ic_all) <= 0.05, "results/phase3_ic.csv", f"max trailing IC {max(tr_ic_all):.3f}"),
    ("negative on three holdout pairs", sum(icv(s, "f_trailing->lambda", -1) < 0 for s in HOLD) == 3, "results/phase3_ic.csv", "trailing λ IC < 0"),
    ("t above 2 on all four held-out pairs", all(min(f1(s, "dm_har_vs_trailing_cal"), f1(s, "dm_har_vs_seasonal_cal")) > 2 for s in HOLD), "results/phase2_forecast.csv", "holdout DM t"),
    ("DM against the seasonal model ranges from 3.7 to 60", round(dev_dm.min(), 1) == 3.7 and round(dev_dm.max()) == 60, "results/phase2_forecast.csv", f"dev configs, min {dev_dm.min():.2f}, max {dev_dm.max():.1f}"),
    ("above 3 in 58 of 60", int((fc[fc.symbol.isin(["BTCUSDT", "BNBUSDT"])]["dm_har_vs_trailing_cal"] > 3).sum()) == 58, "results/phase2_forecast.csv", "dev configs vs trailing"),
    ("IC 0.08 to 0.12, t above 16", round(min(vol_ic), 2) == 0.08 and round(max(vol_ic), 2) == 0.12 and min(vol_t) > 16, "results/phase3_ic.csv", f"vol IC {min(vol_ic):.3f}-{max(vol_ic):.3f}, min t {min(vol_t):.1f}"),
    ("on 5 of 6 pairs", oracle_worse == 5, "results/phase4_policies.csv", "oracle below always-on, avoid_high τ=60 κ=1 fee 0"),
    ("−0.043 bps, 95% CI −0.083 to −0.003", round(sc[sc.symbol == "LTCBTC"]["P6_diff_bps"].iloc[0], 3) == -0.043, "results/scorecard.csv", "LTCBTC P6"),
    ("BTCUSDT, BNBUSDT, ETHBTC and LTCBTC", sig == ["BTCUSDT", "BNBUSDT", "ETHBTC", "LTCBTC"], "results/phase4_execution.csv", "CI upper bound < 0 at rate 5%"),
    ("about 1 to 2% of cost", all(1.0 <= round(pct_save[s], 1) <= 2.0 for s in ["BTCUSDT", "ETHBTC", "LTCBTC"]), "results/phase4_execution.csv", ", ".join(f"{s} {pct_save[s]:.2f}%" for s in ["BTCUSDT", "ETHBTC", "LTCBTC"])),
    ("8% on BNBUSDT", round(pct_save["BNBUSDT"]) == 8, "results/phase4_execution.csv", f"{pct_save['BNBUSDT']:.2f}%"),
    ("2 of 4", holdout_pass["P7"] == 2, "results/scorecard.csv", "P7 holdout passes"),
    ("from 1 of 4 to 2 of 4", asrun_pass["P7"] == 1 and holdout_pass["P7"] == 2, "results/scorecard_as_run.csv, scorecard.csv", "P7 as-run vs corrected"),
    ("P6 failed before and after", asrun_pass["P6"] == 0 and holdout_pass["P6"] == 0, "results/scorecard_as_run.csv, scorecard.csv", "P6 0/4 both"),
    ("differ by 9%", round(100 * (1 - ex("QTUMUSDT").volume_ratio)) == 9, "results/phase4_execution.csv", "QTUMUSDT volume ratio vs trailing"),
    ("−53% to +222%", round(-100 * ex("QTUMUSDT").ci_hi / ex("QTUMUSDT").cost_b_bps) == -53 and round(-100 * ex("QTUMUSDT").ci_lo / ex("QTUMUSDT").cost_b_bps) == 222, "results/phase4_execution.csv", "QTUMUSDT saving CI"),
    ("0.78 at 5 minutes, against 0.57", round(ven[(ven.venue == "AMZN") & (ven.window_sec == 300) & (ven.estimator == "ofi")]["reliability_sb"].iloc[0], 2) == 0.78
     and round(ven[(ven.venue == "AMZN") & (ven.window_sec == 300) & (ven.estimator == "kyle_usd")]["reliability_sb"].iloc[0], 2) == 0.57, "results/phase5_venues_reliability.csv", "AMZN OFI / Kyle"),
    ("1,588 and 734 trades", set(ven[ven.venue.isin(["CB_BTC", "CB_ETH"])]["trades"]) == {1588, 734}, "results/phase5_venues_reliability.csv", "Coinbase trade counts"),
    ("44.2", round(rel[(rel.symbol == "BTCUSDT") & (rel.estimator == "kyle_usd") & (rel.bar_sec == 1) & (rel.window_sec == 300) & (rel.split == "all")]["median_lambda"].iloc[0], 1) == 44.2, "results/phase1_reliability.csv", "BTCUSDT median 5-min λ"),
    ("8f3e63b", "8f3e63b" in commits, "git log", commits.get("8f3e63b", "missing")),
    ("25 Sep 18:33:59", commits.get("8f3e63b", "").startswith("2026-09-25T18:33:59"), "git log", "freeze commit time, UTC"),
    ("25 Sep 18:34:27", commits.get("2f28695", "").startswith("2026-09-25T18:34:27"), "git log", "loader commit time, UTC"),
    ("25 Sep 19:03:28", commits.get("a2b39e8", "").startswith("2026-09-25T19:03:28"), "git log", "holdout results commit time, UTC"),
    ("28 seconds later", commits.get("8f3e63b", "")[11:19] == "18:33:59" and commits.get("2f28695", "")[11:19] == "18:34:27", "git log", "18:34:27 minus 18:33:59"),
    ("Constancy rejected far more often than chance on 3 of 4 pairs", holdout_pass["P1"] == 3, "results/scorecard.csv", "P1"),
    ("on 4 of 4 pairs", holdout_pass["P3"] == 4, "results/scorecard.csv", "P3"),
    ("0 of 4 pairs", holdout_pass["P4"] == 0 and holdout_pass["P6"] == 0, "results/scorecard.csv", "P4, P6"),
]

QUALITATIVE = [
    ("Predictions and thresholds were committed before any holdout analysis", "`docs/PREREGISTRATION.md` in commit `8f3e63b`; holdout results first appear in `a2b39e8`."),
    ("Holdout bars were built before the freeze, but no analysis ran on them", "`docs/PREREGISTRATION.md` intro and command list; bar build is the only pre-freeze step."),
    ("Loader modules were left out of the freeze by `.gitignore` and committed unchanged before the holdout", "`docs/PREREGISTRATION.md`, Deviations log; commit `2f28695`."),
    ("The units bug, its fix, a regression test, both scorecards kept, thresholds unchanged", "`src/kylelambda/markout.py` (`fx`), `tests/test_bars_markout.py::test_markout_half_spread_with_fx_conversion`, `results/scorecard_as_run.csv`, Deviations log."),
    ("Baselines were first handicapped, then clipped and calibrated the same way as the model", "`src/kylelambda/forecast.py` (`trailing_cal`, `seasonal_cal`, clipped raw baselines); only the corrected results are in `results/phase2_forecast.csv`."),
    ("The execution schedules were normalized to equal expected volume", "`src/kylelambda/policy.py::run_execution` (tilt renormalization); `volume_ratio` column in `results/phase4_execution.csv`."),
    ("Kyle solver matches the closed forms", "`tests/test_kyle.py`."),
    ("The websocket collector was never run live; no result uses it", "`src/kylelambda/collector/ws.py`, `tests/test_collector.py`; no `data/ws/` input is read by any script."),
    ("The one-day chart's date was picked by rule", "`scripts/make_figures.py::headline` and `site/scripts/export_site_data.py::headline_day`: max realized variance in 2019 with 288 valid windows."),
    ("The impact slider is an illustration", "`site/src/charts/impact.ts`: fixed λ = 20 and 80, labelled as illustrative on the page."),
    ("Exploratory observations were not tested on the holdout", "Maker realized-spread IC and the reversed quoting rule appear only in development analysis (README, Phase 3 and 4a); neither is a scored prediction."),
    ("Nearby literature exists; no novelty claim", "Goyenko, Holden and Trzcinka (2009), *JFE* 92(2); Easley, López de Prado and O'Hara (2012), *RFS* 25(5)."),
    ("I did not collect the data myself; I used AI coding tools", "`data/README.md` lists every public source. Stated on the page under My role."),
]


def main():
    failures = []
    rows = []
    for phrase, ok, src, detail in CLAIMS:
        on_page = phrase in TEXT
        good = bool(ok) and on_page
        if not good:
            failures.append((phrase, bool(ok), on_page))
        rows.append(f"| {phrase.replace('|', '/')} | {src} | {detail} | {'pass' if good else 'FAIL'} |")
    tests_ok = str(n_tests) in TEXT and n_tests == DATA["meta"]["tests"]
    rows.append(f"| {n_tests} automated tests | pytest --collect-only | {n_tests} collected | {'pass' if tests_ok else 'FAIL'} |")
    if not tests_ok:
        failures.append(("tests", n_tests, DATA["meta"]["tests"]))

    audit = ROOT / "site" / "ACCURACY_AUDIT.md"
    audit.write_text("\n".join([
        "# Accuracy audit",
        "",
        "Every claim on the project website, mapped to the file that supports it. The numeric table is",
        "generated by `site/scripts/verify_claims.py`, which recomputes each value from `results/`, git and",
        "pytest, and checks that the exact phrase appears in `site/index.html`. Chart values are not typed",
        "into the page at all: `site/scripts/export_site_data.py` exports them to `site/src/data/site-data.json`.",
        "",
        "## Numeric claims (automated)",
        "",
        "| Phrase on the page | Source | Recomputed | Check |",
        "|---|---|---|---|",
        *rows,
        "",
        "## Process and scope claims (manual)",
        "",
        "| Claim | Evidence |",
        "|---|---|",
        *[f"| {c} | {e} |" for c, e in QUALITATIVE],
        "",
        "## Claims deliberately qualified or left out",
        "",
        "* **\"Holdout lambda IC 0.10 to 0.26.\"** 0.26 is BNBUSDT, a development pair. The page says 0.10 to 0.17 on the",
        f"  holdout; across all six pairs it is {min(lam_ic_all):.2f} to {max(lam_ic_all):.2f}.",
        "* **\"1 to 2% saving on liquid pairs\"** is true for BTCUSDT, ETHBTC and LTCBTC. BNBUSDT saved about 8%, and the",
        "  page says so next to the range.",
        "* **AMZN OFI reliability is 0.78, not 0.79.** The unrounded value is 0.7849. An earlier README table and the",
        "  project brief said 0.79 (a double rounding); this check caught it and both were corrected.",
        "* **Oracle quote gating** was worse than always quoting on 5 of 6 pairs, not all six. The exception (QTUMUSDT) is named.",
        "* **\"Constant lambda rejected on 27 to 53% of days\"** is the development pairs only. Holdout pairs are reported",
        "  separately (7 to 31%).",
        "* **Coinbase L3** is presented as insufficient data, not as evidence either way. **AMZN** is one day and is",
        "  described as consistent with the crypto results, not as equity evidence.",
        "* **No live trading, no deployment, no live data collection** is claimed anywhere.",
        "",
    ]) + "\n")
    print(f"{len(CLAIMS) + 1} numeric checks, {len(failures)} failures; wrote {audit.relative_to(ROOT)}")
    for f in failures:
        print("  FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
