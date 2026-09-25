"""Figures for the README. Static PNGs (GitHub renders them inline).

    python scripts/make_figures.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kylelambda.io import DATA, FIG, RESULTS, load_windows  # noqa: E402
from kylelambda.kyle import kyle_equilibrium  # noqa: E402
from kylelambda.signal import load_panel  # noqa: E402

# reference palette, light mode: categorical slots 1-3, text and grid inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def headline():
    """One day of BTCUSDT: noisy 5-min lambda, Kyle's constant, and the forecastable part."""
    sym, W = "BTCUSDT", 300
    p = load_panel(sym, "kyle_usd", W, 1)
    w = load_windows(sym, 1, W).set_index("t0")
    # the 2019 day with the largest realized variance and no masked windows; picked by rule, not by eye
    daily = w.groupby("date").agg(rv=("rv_bps2", "sum"), n=("kyle_usd", "count"))
    cand = daily[(daily.index >= "2019-01-01") & (daily.n == 288)]
    date = cand["rv"].idxmax()
    t0 = pd.Timestamp(date).value // 10**9
    idx = np.arange(t0, t0 + 86400, W)
    lam = w["kyle_usd"].reindex(idx)
    fc = p["f_har"].reindex(idx - W)  # forecast made one window earlier for window idx
    # the constant a desk would carry: trailing 7-day median of the 5-min estimate (z = 1 in phase 2)
    const = w["kyle_usd"].loc[:t0 - 1].iloc[-7 * 288:].median()
    x = pd.to_datetime(idx, unit="s")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(x, const, fc.to_numpy(), color=BLUE, alpha=0.18, linewidth=0, label="predictable deviation from the constant")
    ax.scatter(x, lam, s=10, color=MUTED, alpha=0.8, linewidths=0, label="5-min estimate (≈57% noise)")
    ax.axhline(const, color=INK, linewidth=1.5, linestyle=(0, (5, 3)), label="Kyle constant (trailing 7-day median λ)")
    ax.plot(x, fc, color=BLUE, linewidth=2, label="forecast λ, made 5 min ahead")
    lo, hi = np.nanquantile(lam, [0.02, 0.98])
    ax.set_ylim(min(lo, 0), hi * 1.1)
    ax.set_ylabel("price impact, bps per $1M signed flow")
    ax.set_title(f"Kyle's lambda is not a constant: BTCUSDT, {date} (UTC)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    save(fig, "fig1_headline.png")
    return date


def reliability_grid():
    r = pd.read_csv(RESULTS / "phase1_reliability.csv")
    r = r[(r.split == "all") & (r.symbol == "BTCUSDT")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, est, title in [(axes[0], "kyle_usd", "Kyle λ (signed $)"), (axes[1], "sqrt_usd", "√-flow λ (Hasbrouck)")]:
        d = r[r.estimator == est].pivot(index="bar_sec", columns="window_sec", values="reliability_sb")
        im = ax.imshow(d.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(d.shape[1]), [{60: "1m", 300: "5m", 900: "15m", 3600: "1h", 14400: "4h", 86400: "1d"}[c] for c in d.columns])
        ax.set_yticks(range(d.shape[0]), [f"{b}s" for b in d.index])
        ax.grid(False)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
        ax.set_title(title)
        ax.set_xlabel("estimation window")
    axes[0].set_ylabel("bar size")
    fig.suptitle("Share of a rolling λ's variance that is real (split-half reliability), BTCUSDT", x=0.01, ha="left",
                 y=1.04, fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.outline.set_visible(False)
    save(fig, "fig2_reliability.png")


def forecast_horizons():
    f = pd.read_csv(RESULTS / "phase2_forecast.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sym in zip(axes, ["BTCUSDT", "BNBUSDT"]):
        d = f[(f.symbol == sym) & (f.estimator == "kyle_usd") & (f.window_sec == 300)].sort_values("h")
        mins = d["horizon_sec"] / 60
        for col, c, lab in [("r2_har", BLUE, "conditional (HAR)"), ("r2_trailing_cal", ORANGE, "trailing λ, calibrated"),
                            ("r2_seasonal_cal", AQUA, "seasonal, calibrated")]:
            ax.plot(mins, 100 * d[col], color=c, marker="o", markersize=5, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 15, 30, 60, 120], ["5m", "10m", "15m", "30m", "1h", "2h"])
        ax.set_title(sym)
        ax.set_xlabel("forecast horizon")
        rel = d["reliability_sb"].iloc[0]
        ax.text(0.98, 0.95, f"noise ceiling ≈ {100 * rel:.0f}%", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=9)
    axes[0].set_ylabel("out-of-sample R² vs Kyle constant, %")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=9)
    fig.suptitle("5-minute λ is forecastable, and not just from time-of-day", x=0.01, y=1.04, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig3_forecast.png")


def signal_ic():
    ic = pd.read_csv(RESULTS / "phase3_ic.csv")
    d = ic[(ic.lead == 1) & (ic.tau.isin([-1, 60]))]
    rows = [("f_har->lambda", "forecast λ → realized λ"), ("f_trailing->lambda", "trailing λ → realized λ"),
            ("f_har", "forecast λ → maker adverse sel."), ("f_rv", "trailing vol → maker adverse sel.")]
    syms = [s for s in ["BTCUSDT", "BNBUSDT", "ETHBTC", "LTCBTC", "NEOUSDT", "QTUMUSDT"] if s in set(d.symbol)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.8 / len(rows)
    cols = [BLUE, "#9dc3ef", ORANGE, AQUA]
    for k, ((pred, lab), c) in enumerate(zip(rows, cols)):
        v = [d[(d.symbol == s) & (d.predictor == pred)]["ic_daily"].mean() for s in syms]
        ax.bar(np.arange(len(syms)) + (k - 1.5) * width, v, width=width * 0.92, color=c, label=lab)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(syms)), syms)
    ax.set_ylabel("mean daily rank IC")
    ax.set_title("The forecast predicts λ itself; it is volatility, not λ, that predicts maker losses")
    ax.legend(ncols=2, fontsize=9, loc="upper right")
    save(fig, "fig4_signal.png")


def execution():
    e = pd.read_csv(RESULTS / "phase4_execution.csv")
    d = e[(e.rate == 0.05) & (e.a == "har") & (e.b.isin(["trailing", "twap"]))]
    syms = [s for s in ["BTCUSDT", "BNBUSDT", "ETHBTC", "LTCBTC", "NEOUSDT", "QTUMUSDT"] if s in set(d.symbol)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    top = 20.0
    for k, (b, c, lab) in enumerate([("twap", MUTED, "vs TWAP (λ assumed constant)"), ("trailing", BLUE, "vs trailing-λ schedule")]):
        for i, s in enumerate(syms):
            r = d[(d.symbol == s) & (d.b == b)].iloc[0]
            base = r["cost_b_bps"]
            y, lo, hi = -100 * r["diff_bps"] / base, -100 * r["ci_hi"] / base, -100 * r["ci_lo"] / base
            x = i + (k - 0.5) * 0.3
            if hi - lo > 40:  # too thin to say anything: annotate instead of letting it set the scale
                if k == 1:
                    ax.text(i, 0.6, f"too thin to estimate\n(CI {lo:.0f}% to {hi:.0f}%)", ha="center", fontsize=8.5, color=INK2)
                continue
            ax.bar(x, y, width=0.28, color=c, label=lab if i == 0 else None)
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axvline(1.5, color=GRID, linewidth=1.5)
    ax.text(0.5, top * 0.93, "development", ha="center", color=INK2, fontsize=9)
    ax.text(3.5, top * 0.93, "holdout (pre-registered)", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(-3, top)
    ax.set_xticks(range(len(syms)), syms)
    ax.set_ylabel("cost saved, % of baseline cost")
    ax.set_title("Kyle-optimal execution on forecast λ, 5% participation (95% day-bootstrap CI)")
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.88))
    save(fig, "fig5_execution.png")


def kyle_anchor():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for N, c in [(4, AQUA), (20, ORANGE), (200, BLUE)]:
        e = kyle_equilibrium(N)
        t = (np.arange(N) + 1) / N
        ax.step(t, e.lam, where="post", color=c, label=f"N = {N} auctions")
    ax.axhline(1.0, color=INK, linewidth=1.2, linestyle=(0, (5, 3)))
    ax.text(0.02, 1.02, "continuous-time limit: λ = σᵥ/σᵤ", color=INK2, fontsize=9, va="bottom")
    ax.set_xlabel("time within the trading period")
    ax.set_ylabel("λ (units of σᵥ/σᵤ)")
    ax.set_title("The anchor: Kyle (1985) equilibrium λ flattens to a constant as trading becomes continuous")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "fig0_kyle.png")


if __name__ == "__main__":
    kyle_anchor()
    print("headline day:", headline())
    reliability_grid()
    forecast_horizons()
    signal_ic()
    execution()
