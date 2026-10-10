from datetime import datetime, timedelta, timezone

from scripts.run_momentum_volatility_research import Bar
from scripts.run_volume_volatility_regime_reversal import candidate, features


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
            Bar(start + timedelta(hours=i), close, high, low, close, volume)
        )
    return rows


def test_reversal_candidate_preserves_frozen_regime_gate():
    data = features(_bars())
    selected = candidate(25, data, data)
    assert selected is not None
    assert selected["direction"] == 1


def test_reversal_requires_the_same_breakout_gate():
    bars = _bars()
    bars[25] = Bar(
        bars[25].timestamp, 100.0, 101.0, 99.0, 100.0, 250.0
    )
    assert candidate(25, features(bars), features(bars)) is None
