"""Websocket recorder for public order-flow feeds.

    python -m kylelambda.collector.ws --venue coinbase --symbols BTC-USD ETH-USD --out data/ws

What is actually L3 on a free public socket (as of writing):

    coinbase   "full" channel: every order add / change / match / done, with sequence numbers
    bitstamp   "live_orders_<pair>": every order created / changed / deleted, plus live_trades

The others below are L2 diffs plus trades. They support the trade-based lambda and
OFI at the touch, not order-level queue analysis:

    binance    <sym>@depth@100ms + <sym>@trade
    bybit      orderbook.200.<SYM> + publicTrade.<SYM>

Design, since a recorder is only useful if you can trust what it did not see:

  * raw messages go to disk untouched (gzipped JSON lines, one file per venue-hour),
    each prefixed with the local receive timestamp in ns; parsing happens offline
  * reconnect with capped exponential backoff and jitter; every (re)connect writes a
    marker line so gaps are explicit in the file, never silent
  * Coinbase sequence numbers are checked per product; a gap writes a GAP marker, and
    the offline parser must re-snapshot from REST before trusting the book again
  * no business logic in the hot path: receive, stamp, write
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("collector")

URLS = {
    "coinbase": "wss://ws-feed.exchange.coinbase.com",
    "bitstamp": "wss://ws.bitstamp.net",
    "binance": "wss://stream.binance.com:9443/stream",
    "bybit": "wss://stream.bybit.com/v5/public/spot",
}


def subscriptions(venue: str, symbols: list[str]) -> list[dict]:
    if venue == "coinbase":
        return [{"type": "subscribe", "product_ids": symbols, "channels": ["full", "heartbeat"]}]
    if venue == "bitstamp":
        subs = []
        for s in symbols:
            for ch in ("live_orders_", "live_trades_"):
                subs.append({"event": "bts:subscribe", "data": {"channel": ch + s.lower()}})
        return subs
    if venue == "binance":
        params = [f"{s.lower()}@{k}" for s in symbols for k in ("depth@100ms", "trade")]
        return [{"method": "SUBSCRIBE", "params": params, "id": 1}]
    if venue == "bybit":
        args = [f"{k}.{s.upper()}" for s in symbols for k in ("orderbook.200", "publicTrade")]
        return [{"op": "subscribe", "args": args}]
    raise ValueError(venue)


@dataclass
class SequenceTracker:
    """Detects dropped messages on feeds with per-product sequence numbers (Coinbase)."""
    last: dict[str, int] = field(default_factory=dict)

    def check(self, msg: dict) -> tuple[str, int, int] | None:
        seq, pid = msg.get("sequence"), msg.get("product_id")
        if seq is None or pid is None:
            return None
        prev = self.last.get(pid)
        self.last[pid] = seq
        if prev is not None and seq != prev + 1 and seq > prev:
            return pid, prev, seq
        return None


class HourlyWriter:
    """Append-only gzipped JSONL, rotated every UTC hour."""

    def __init__(self, root: Path, venue: str):
        self.root, self.venue = root, venue
        self._hour = None
        self._fh = None

    def _rotate(self, now: float):
        hour = int(now // 3600)
        if hour != self._hour:
            if self._fh:
                self._fh.close()
            ts = time.strftime("%Y-%m-%dT%H", time.gmtime(hour * 3600))
            path = self.root / self.venue / f"{ts}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(path, "at")
            self._hour = hour

    def write(self, recv_ns: int, raw: str):
        self._rotate(recv_ns / 1e9)
        self._fh.write(f"{recv_ns}\t{raw}\n")

    def marker(self, kind: str, **info):
        now = time.time_ns()
        self.write(now, json.dumps({"_marker": kind, **info}))
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


async def record(venue: str, symbols: list[str], out: Path, max_backoff: float = 60.0,
                 stop_after: float | None = None):
    import websockets  # optional dependency: pip install .[collector]

    writer = HourlyWriter(out, venue)
    seq = SequenceTracker()
    backoff = 1.0
    t_end = time.time() + stop_after if stop_after else None
    try:
        while t_end is None or time.time() < t_end:
            try:
                async with websockets.connect(URLS[venue], ping_interval=20, max_size=2 ** 24) as ws:
                    for sub in subscriptions(venue, symbols):
                        await ws.send(json.dumps(sub))
                    writer.marker("connect", symbols=symbols)
                    backoff = 1.0
                    async for raw in ws:
                        now = time.time_ns()
                        writer.write(now, raw)
                        if venue == "coinbase":
                            gap = seq.check(json.loads(raw))
                            if gap:
                                writer.marker("gap", product=gap[0], prev=gap[1], seq=gap[2])
                                log.warning("sequence gap %s %d -> %d", *gap)
                        if t_end and time.time() >= t_end:
                            break
            except (OSError, asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 - log and reconnect
                writer.marker("disconnect", error=repr(e))
                delay = min(max_backoff, backoff) * (0.5 + random.random())
                log.warning("%s disconnected (%r); reconnecting in %.1fs", venue, e, delay)
                await asyncio.sleep(delay)
                backoff *= 2
    finally:
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=sorted(URLS), required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/ws"))
    ap.add_argument("--hours", type=float, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(record(a.venue, a.symbols, a.out, stop_after=a.hours * 3600 if a.hours else None))


if __name__ == "__main__":
    main()
