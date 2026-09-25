"""Book-level data for the cross-venue phase: LOBSTER (Nasdaq equities) and Coinbase L3.

Both are reduced to the same two streams:

    quotes  time, bid_px, bid_sz, ask_px, ask_sz   (after every book event)
    trades  time, price, qty, sign                 (sign = aggressor side)

and then to the same one-second bar schema as the Binance tape (bars.py), plus the
Cont-Kukanov-Stoikov (2014) order-flow imbalance at the best quotes. With the book
available, the mid is the true mid, not a bounce-corrected proxy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict


# ---------------------------------------------------------------- LOBSTER
def load_lobster(message_csv: Path, orderbook_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LOBSTER message + level-1 book. Prices are in 1e-4 dollars; times in seconds after midnight.

    Execution messages (types 4 visible, 5 hidden) carry the direction of the *resting*
    order, so the aggressor sign is its negative.
    """
    msg = pd.read_csv(message_csv, header=None, names=["time", "type", "oid", "size", "price", "direction"],
                      usecols=range(6))
    ob = pd.read_csv(orderbook_csv, header=None, usecols=[0, 1, 2, 3], names=["ask_px", "ask_sz", "bid_px", "bid_sz"])
    q = pd.DataFrame({"time": msg["time"], "bid_px": ob["bid_px"] / 1e4, "bid_sz": ob["bid_sz"].astype(float),
                      "ask_px": ob["ask_px"] / 1e4, "ask_sz": ob["ask_sz"].astype(float)})
    # LOBSTER marks empty sides with dummy prices (+-9999999999); treat as missing
    bad = (q["ask_px"] > 1e5) | (q["bid_px"] <= 0) | (q["bid_px"] >= q["ask_px"])
    q.loc[bad, ["bid_px", "ask_px"]] = np.nan
    ex = msg[msg["type"].isin([4, 5])]
    trades = pd.DataFrame({"time": ex["time"].to_numpy(), "price": ex["price"].to_numpy() / 1e4,
                           "qty": ex["size"].to_numpy().astype(float), "sign": -ex["direction"].to_numpy()})
    return q, trades


# ---------------------------------------------------------------- Coinbase L3
def _secs(s: pd.Series) -> np.ndarray:
    hh = s.str.slice(0, 2).astype(int); mm = s.str.slice(3, 5).astype(int); ss = s.str.slice(6).astype(float)
    return (hh * 3600 + mm * 60 + ss).to_numpy()


def reconstruct_coinbase_l3(snapshot: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replay a CoinAPI-normalised Coinbase full-channel feed into best quotes and signed trades.

    ADD inserts an order, DELETE removes it, MATCH reduces the resting (maker) order by
    the traded size, SET overwrites an order's size. The MATCH row carries the maker's
    order id and side (checked: it is always an order already on the book, on the side
    given by is_buy), so the aggressor is the opposite side.
    """
    orders: dict[str, list] = {}
    levels = {1: SortedDict(), 0: SortedDict()}

    def add_level(side, px, sz):
        lv = levels[side]
        v = lv.get(px, 0.0) + sz
        if v <= 1e-12:
            lv.pop(px, None)
        else:
            lv[px] = v

    for r in snapshot.itertuples(index=False):
        orders[r.order_id] = [r.is_buy, r.entry_px, r.entry_sx]
        add_level(r.is_buy, r.entry_px, r.entry_sx)

    t = _secs(events["time_exchange"])
    typ = events["update_type"].to_numpy()
    oid = events["order_id"].to_numpy()
    side = events["is_buy"].to_numpy()
    px = events["entry_px"].to_numpy()
    sz = events["entry_sx"].to_numpy()
    qrows, trows = [], []
    stats = {"unknown_order": 0, "crossed": 0, "events": len(events)}
    for i in range(len(events)):
        k = typ[i]
        if k == "ADD":
            orders[oid[i]] = [side[i], px[i], sz[i]]
            add_level(side[i], px[i], sz[i])
        elif k in ("DELETE", "MATCH", "SET"):
            o = orders.get(oid[i])
            if o is None:
                stats["unknown_order"] += 1
            elif k == "DELETE":
                add_level(o[0], o[1], -o[2])
                del orders[oid[i]]
            elif k == "MATCH":
                fill = min(sz[i], o[2])
                add_level(o[0], o[1], -fill)
                o[2] -= fill
                if o[2] <= 1e-12:
                    del orders[oid[i]]
                trows.append((t[i], px[i], sz[i], -1 if o[0] == 1 else 1))
            else:  # SET: new size for an existing order
                add_level(o[0], o[1], sz[i] - o[2])
                o[2] = sz[i]
        # emit L1 once per timestamp (after the last event sharing it)
        if i + 1 == len(events) or t[i + 1] != t[i]:
            b, a = levels[1], levels[0]
            if len(b) and len(a):
                bp, bs = b.peekitem(-1)
                ap, as_ = a.peekitem(0)
                if bp >= ap:
                    stats["crossed"] += 1
                    bp = ap = np.nan
                qrows.append((t[i], bp, bs, ap, as_))
    q = pd.DataFrame(qrows, columns=["time", "bid_px", "bid_sz", "ask_px", "ask_sz"])
    tr = pd.DataFrame(trows, columns=["time", "price", "qty", "sign"])
    return q, tr, stats


# ---------------------------------------------------------------- common bars
def ofi(q: pd.DataFrame) -> np.ndarray:
    """Cont, Kukanov & Stoikov (2014) order-flow imbalance, one value per quote update."""
    bp, bs, ap, as_ = (q[c].to_numpy() for c in ["bid_px", "bid_sz", "ask_px", "ask_sz"])
    bp0, bs0, ap0, as0 = (np.r_[np.nan, x[:-1]] for x in (bp, bs, ap, as_))
    with np.errstate(invalid="ignore"):
        e = (np.where(bp >= bp0, bs, 0) - np.where(bp <= bp0, bs0, 0)
             - np.where(ap <= ap0, as_, 0) + np.where(ap >= ap0, as0, 0))
    e[~np.isfinite(e)] = 0.0
    return e


def bars_from_l1(q: pd.DataFrame, trades: pd.DataFrame, t_start: int, t_end: int) -> pd.DataFrame:
    """One-second bars with the same columns as bars.build_second_bars, plus ofi / ofi_usd."""
    n = t_end - t_start
    q = q[(q["time"] >= t_start) & (q["time"] < t_end)].reset_index(drop=True)
    tr = trades[(trades["time"] >= t_start) & (trades["time"] < t_end)].reset_index(drop=True)
    mid_q = (q["bid_px"] + q["ask_px"]) / 2
    sec_q = (q["time"] - t_start).astype(int).to_numpy()
    e = ofi(q)
    # group fills at one timestamp and side into one taker order (a sweep)
    key = (tr["time"].astype(str) + tr["sign"].astype(str)).to_numpy()
    brk = np.r_[True, key[1:] != key[:-1]]
    gid = np.cumsum(brk) - 1
    od = pd.DataFrame({"time": tr["time"], "sign": tr["sign"], "qty": tr["qty"],
                       "notional": tr["price"] * tr["qty"]}).groupby(gid).agg(
        time=("time", "first"), sign=("sign", "first"), qty=("qty", "sum"), notional=("notional", "sum"))
    sec = (tr["time"] - t_start).astype(int).to_numpy()
    osec = (od["time"] - t_start).astype(int).to_numpy()
    sg = tr["sign"].to_numpy().astype(float)
    osg = od["sign"].to_numpy().astype(float)

    def acc(idx, w):
        return np.bincount(idx, weights=w, minlength=n)[:n]

    last = pd.Series(np.arange(len(sec_q))).groupby(sec_q).max()
    mid = np.full(n, np.nan)
    mid[last.index] = mid_q.to_numpy()[last.to_numpy()]
    mid = pd.Series(mid).ffill().bfill().to_numpy()
    spread = np.full(n, np.nan)
    spread[last.index] = (q["ask_px"] - q["bid_px"]).to_numpy()[last.to_numpy()]
    mid_at_q = mid[np.clip(sec_q, 0, n - 1)]
    bars = pd.DataFrame({
        "n_fills": np.bincount(sec, minlength=n)[:n].astype(np.int32),
        "n_orders": np.bincount(osec, minlength=n)[:n].astype(np.int32),
        "n_buy": acc(osec, (osg > 0).astype(float)),
        "n_sell": acc(osec, (osg < 0).astype(float)),
        "vol": acc(sec, tr["qty"].to_numpy()),
        "notional": acc(sec, (tr["price"] * tr["qty"]).to_numpy()),
        "sv": acc(sec, sg * tr["qty"].to_numpy()),
        "sn": acc(sec, sg * (tr["price"] * tr["qty"]).to_numpy()),
        "ssqrt": acc(osec, osg * np.sqrt(od["qty"].to_numpy())),
        "ssqrt_usd": acc(osec, osg * np.sqrt(od["notional"].to_numpy())),
        "mid": mid,
        "last_px": mid,
        "spread_hat": pd.Series(spread).ffill().bfill().to_numpy(),
        "ofi": acc(sec_q, e),
        "ofi_usd": acc(sec_q, e * mid_at_q),
        "n_quotes": np.bincount(sec_q, minlength=n)[:n],
    })
    bars.index = pd.RangeIndex(t_start, t_end, name="t")
    return bars
