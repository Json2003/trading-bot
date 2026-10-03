from datetime import datetime, timedelta, timezone

from scripts.run_momentum_volatility_research import Bar
from scripts.run_volume_volatility_regime_hypothesis import (
    HOLD_BARS,
    candidate,
    collect_segment,
    features,
)


def _bars(count=30):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(count):
        close = 100.0
        high = 101.0
        low = 99.0
        volume = 100.0
        if i == 25:
            close = 103.0
            high = 104.0
            low = 99.0
            volume = 250.0
        rows.append(
            Bar(
                start + timedelta(hours=i),
                close,
                high,
                low,
                close,
                volume,
            )
        )
    return rows


def test_candidate_requires_both_volume_range_and_breakout():
    bars = _bars()
    data = features(bars)
    selected = candidate(25, data, data)
    assert selected is not None
    assert selected["direction"] == 1
    assert selected["volume_ratio"] >= 2.0
    assert selected["range_ratio"] >= 1.5


def test_no_candidate_without_breakout():
    bars = _bars()
    bars[25] = Bar(
        bars[25].timestamp,
        100.0,
        101.0,
        99.0,
        100.0,
        250.0,
    )
    data = features(bars)
    assert candidate(25, data, data) is None


def test_collect_segment_uses_non_overlapping_windows():
    bars = _bars(80)
    data = features(bars)
    class Pair:
        def __init__(self, bar):
            self.timestamp = bar.timestamp
            self.btc = bar
            self.eth = bar
    pair = [Pair(bar) for bar in bars]
    rows = collect_segment(pair, data, data, 25, len(pair))
    assert all(
        b["signal_index"] - a["signal_index"] > HOLD_BARS
        for a, b in zip(rows, rows[1:])
    )
