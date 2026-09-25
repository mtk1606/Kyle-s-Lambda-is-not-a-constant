"""Adverse selection on passive fills, from the one-second bars.

For a fill i at second s with aggressor sign d_i, notional N_i and pre-trade mid m_{s-1}:

    effective half-spread earned by the maker   HS_i = d_i (p_i - m_{s-1}) / m_{s-1}
    adverse selection at horizon tau            AS_i = d_i (m_{s+tau} - m_{s-1}) / m_{s-1}
    maker realized spread                       RS_i = HS_i - AS_i

The aggressor's direction is the maker's loss, so AS > 0 means the maker got picked
off. Both sums only need per-second aggregates:

    sum_i N_i HS_i ~= sn_s - m_{s-1} sv_s          (p_i / m_{s-1} ~= 1)
    sum_i N_i AS_i = sn_s (m_{s+tau} / m_{s-1} - 1)

so I never have to touch individual trades again. Window values are notional-weighted
means in bps. Fills whose horizon runs past the end of the day are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import valid_seconds

TAUS = (1, 10, 60, 300)


def window_markouts(bars1s: pd.DataFrame, window_sec: int, taus=TAUS) -> pd.DataFrame:
    n = len(bars1s)
    per = window_sec // 1
    nwin = n // per
    m = bars1s["mid"].to_numpy()
    m_prev = np.r_[np.nan, m[:-1]]
    sn = bars1s["sn"].to_numpy()
    sv = bars1s["sv"].to_numpy()
    notional = bars1s["notional"].to_numpy()
    ok = valid_seconds(bars1s)
    out = {"t0": bars1s.index.to_numpy()[: nwin * per : per]}
    # d_i (p_i - m) / m * p_i q_i ~= d_i (p_i - m) q_i, in quote currency
    hs = np.where(np.isfinite(m_prev), sn - m_prev * sv, 0.0)
    base_ok = np.isfinite(m_prev) & ok
    for tau in taus:
        m_fwd = np.r_[m[tau:], np.full(tau, np.nan)]
        ok_fwd = np.r_[ok[tau:], np.zeros(tau, bool)]
        use = base_ok & np.isfinite(m_fwd) & ok_fwd
        a = np.where(use, sn * (m_fwd / m_prev - 1), 0.0)
        w = np.where(use, notional, 0.0)
        A = a[: nwin * per].reshape(nwin, per).sum(1)
        Wt = w[: nwin * per].reshape(nwin, per).sum(1)
        H = np.where(use, hs, 0.0)[: nwin * per].reshape(nwin, per).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"as_{tau}"] = 1e4 * A / Wt
            out[f"hs_{tau}"] = 1e4 * H / Wt
        out[f"notional_{tau}"] = Wt / 1e6
    df = pd.DataFrame(out)
    for tau in taus:
        df.loc[df[f"notional_{tau}"] <= 0, [f"as_{tau}", f"hs_{tau}"]] = np.nan
    return df
