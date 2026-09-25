import gzip
import json

from kylelambda.collector.ws import HourlyWriter, SequenceTracker, subscriptions


def test_sequence_gap_detection():
    s = SequenceTracker()
    assert s.check({"product_id": "BTC-USD", "sequence": 10}) is None
    assert s.check({"product_id": "BTC-USD", "sequence": 11}) is None
    assert s.check({"product_id": "ETH-USD", "sequence": 500}) is None  # products tracked separately
    assert s.check({"product_id": "BTC-USD", "sequence": 14}) == ("BTC-USD", 11, 14)
    assert s.check({"type": "heartbeat"}) is None


def test_writer_rotates_hourly_and_marks(tmp_path):
    w = HourlyWriter(tmp_path, "coinbase")
    w.write(3600 * 10**9 * 5 + 1, '{"a":1}')
    w.write(3600 * 10**9 * 6 + 1, '{"a":2}')
    w.marker("gap", product="BTC-USD")
    w.close()
    files = sorted((tmp_path / "coinbase").glob("*.jsonl.gz"))
    assert len(files) >= 2
    first = gzip.open(files[0], "rt").read().splitlines()
    assert first[0].split("\t")[1] == '{"a":1}'


def test_subscriptions_shape():
    assert subscriptions("coinbase", ["BTC-USD"])[0]["channels"][0] == "full"
    assert "btcusdt@trade" in subscriptions("binance", ["BTCUSDT"])[0]["params"]
    assert json.dumps(subscriptions("bitstamp", ["BTCUSD"]))
