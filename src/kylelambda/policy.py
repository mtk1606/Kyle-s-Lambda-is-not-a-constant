"""Phase 4: does conditioning a quoting rule on forecast lambda pay, net of costs?

The policy. A passive liquidity provider takes a small, fixed share phi of the maker
side of every fill in the windows where it is quoting. Before each window it decides
whether to quote, using a score known at the end of the previous window:

    quote in window t+1  iff  score_t <= q-th percentile of score over the past 7 days
                              (or >= the (1-q)-th percentile for the reversed rule)

so every gated policy is active on the same ~q share of windows and the comparison is
about *which* windows it skips, not how many. Scores compared:

    always     quote everywhere (no gate)
    trailing   the desk practice: EWMA of past lambda estimates
    seasonal   time-of-day lambda profile
    har        the Phase 2 conditional forecast
    oracle     realized lambda in t+1 (not tradeable; an upper bound for scale)

PnL per unit notional of a fill, in bps:

    realized spread - fee = HS - kappa * AS(tau) - fee

HS and AS(tau) are the notional-weighted effective half-spread and adverse selection of
all maker fills in the window (markout.py). kappa >= 1 inflates adverse selection
because a queue-position-blind share of fills is optimistic: in practice the fills you
get are skewed toward the ones that go against you. tau is the horizon at which the
maker is assumed to have hedged or unwound.

Inference is a day-clustered bootstrap on the difference in PnL between two policies,
so intraday correlation and volatility clustering are respected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import DAY


def causal_threshold(score: pd.Series, window_sec: int, q: float, days: int = 7) -> pd.Series:
    n = days * DAY // window_sec
    return score.shift(1).rolling(n, min_periods=n // 3).quantile(q)


def run_policies(p: pd.DataFrame, window_sec: int, q: float = 0.8, tau: int = 60,
                 kappa: float = 1.0, fee_bps: float = 0.0, phi: float = 0.01,
                 direction: str = "avoid_high") -> pd.DataFrame:
    """Per-window PnL (in $) for each policy. p comes from signal.load_panel with h=1.

    direction='avoid_high' is the hypothesis as stated up front: step away when lambda is
    expected to be high (toxic). 'avoid_low' is the reverse, added after Phase 3 showed
    that high expected lambda comes with *better* maker realized spreads.
    """
    edge = p[f"hs_{tau}"] - kappa * p[f"as_{tau}"] - fee_bps
    vol = phi * p["notional_next"] * 1e6
    ok = edge.notna() & vol.notna()
    out = pd.DataFrame(index=p.index)
    out["date"] = p["date"]
    out["always"] = np.where(ok, vol * edge / 1e4, np.nan)
    out["vol_always"] = np.where(ok, vol, np.nan)
    for name, col in [("trailing", "f_trailing"), ("seasonal", "f_seasonal"), ("har", "f_har"),
                      ("oracle", "lambda_next")]:
        s = p[col]
        if direction == "avoid_high":
            on = (s <= causal_threshold(s, window_sec, q)) & ok
        else:
            on = (s >= causal_threshold(s, window_sec, 1 - q)) & ok
        out[name] = np.where(ok, np.where(on, vol * edge / 1e4, 0.0), np.nan)
        out[f"vol_{name}"] = np.where(ok, np.where(on, vol, 0.0), np.nan)
    return out.dropna(subset=["always"])


def summarize(pnl: pd.DataFrame, policies=("always", "trailing", "seasonal", "har", "oracle")) -> pd.DataFrame:
    rows = []
    for k in policies:
        v = pnl[f"vol_{k}"].sum()
        rows.append({"policy": k, "pnl": pnl[k].sum(), "volume": v, "bps_per_notional": 1e4 * pnl[k].sum() / v,
                     "active_share": (pnl[f"vol_{k}"] > 0).mean()})
    return pd.DataFrame(rows).set_index("policy")


def diff_ci(pnl: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0, per: str = "bps") -> tuple:
    """Day-clustered bootstrap CI for (a - b).

    per='bps'  difference in net bps per $ traded
    per='usd'  difference in total PnL, as bps of the always-on volume (so policies that
               trade less are not flattered by a small denominator)
    """
    g = pnl.groupby("date")[[a, b, f"vol_{a}", f"vol_{b}", "vol_always"]].sum()
    arr = g.to_numpy()

    def stat(x):
        if per == "bps":
            return 1e4 * (x[:, 0].sum() / x[:, 2].sum() - x[:, 1].sum() / x[:, 3].sum())
        return 1e4 * (x[:, 0].sum() - x[:, 1].sum()) / x[:, 4].sum()

    rng = np.random.default_rng(seed)
    k = len(arr)
    boots = np.array([stat(arr[rng.integers(0, k, k)]) for _ in range(n_boot)])
    point = stat(arr)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float((boots <= 0).mean())


# ---------------------------------------------------------------------------- execution
"""Execution side. A trader has to buy (or sell) a steady flow of size over the day and
can lean the schedule toward cheap windows. Under Kyle's linear impact a child order of
q $M in window w costs

    c_w(q) = HS_w + lambda_w * q / 2        bps per $ executed

(effective half-spread plus half the linear impact of the order itself), and for a fixed
total the cost-minimizing split is q_w proportional to 1 / lambda_w. Every schedule uses
that rule and differs only in which lambda it plugs in:

    twap       q_w = q_bar                             (lambda assumed constant: Kyle)
    trailing   q_w = q_bar * (1 / trailing_w) / m      (desk practice)
    seasonal   q_w = q_bar * (1 / seasonal_w) / m
    har        q_w = q_bar * (1 / forecast_w) / m      (this project)

m is the trailing 7-day mean of 1/score (so the tilt averages 1), which makes every
schedule trade q_bar per window in expectation without looking ahead. Tilts are clipped to [1/4, 4] and child orders
to 20% of the window's traded notional, identically for all schedules.

Costs are evaluated with the *realized* lambda estimate of window w. Its noise is
independent of anything known at t-1, so it is an unbiased stand-in for the true lambda
in the expected cost; no schedule gets to peek at it.
"""


def run_execution(p: pd.DataFrame, window_sec: int, rate: float, tilt_clip: float = 4.0) -> pd.DataFrame:
    """Child-order costs per window. rate = base child size as a share of the trailing
    7-day median window notional (so the schedule scales with the market, causally)."""
    ok = p["lambda_next"].notna() & p["hs_60"].notna() & p["notional_next"].gt(0)
    lam = p["lambda_next"]
    n7 = 7 * DAY // window_sec
    q_bar = rate * p["notional_next"].shift(1).rolling(n7, min_periods=DAY // window_sec).median()
    ok &= q_bar.notna()
    out = pd.DataFrame(index=p.index)
    out["date"] = p["date"]
    for name, col in [("twap", None), ("trailing", "f_trailing"), ("seasonal", "f_seasonal"), ("har", "f_har")]:
        if col is None:
            tilt = pd.Series(1.0, index=p.index)
        else:
            inv = 1.0 / p[col].where(p[col] > 0)
            med = inv.shift(1).rolling(n7, min_periods=DAY // window_sec).median()
            raw = (inv / med).clip(1 / tilt_clip, tilt_clip)
            # renormalize by the trailing mean of the clipped tilt so every schedule has the
            # same expected volume; otherwise convex impact punishes whoever trades more
            tilt = (raw / raw.shift(1).rolling(n7, min_periods=DAY // window_sec).mean()).fillna(1.0)
        q = q_bar * tilt
        cost = p["hs_60"] + lam * q / 2
        out[name] = np.where(ok, q * cost, np.nan)        # bps * $M
        out[f"vol_{name}"] = np.where(ok, q, np.nan)
    out["vol_always"] = out["vol_twap"]
    return out.dropna(subset=["twap"])


def summarize_execution(ex: pd.DataFrame, policies=("twap", "trailing", "seasonal", "har")) -> pd.DataFrame:
    return pd.DataFrame([{"policy": k, "cost_bps": ex[k].sum() / ex[f"vol_{k}"].sum(),
                          "volume_musd": ex[f"vol_{k}"].sum()} for k in policies]).set_index("policy")


def execution_diff_ci(ex: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Day-clustered bootstrap CI for cost_bps(a) - cost_bps(b). Negative = a is cheaper."""
    g = ex.groupby("date")[[a, b, f"vol_{a}", f"vol_{b}"]].sum().to_numpy()

    def stat(x):
        return x[:, 0].sum() / x[:, 2].sum() - x[:, 1].sum() / x[:, 3].sum()

    rng = np.random.default_rng(seed)
    k = len(g)
    boots = np.array([stat(g[rng.integers(0, k, k)]) for _ in range(n_boot)])
    return stat(g), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float((boots >= 0).mean())
