from datetime import datetime, timedelta, timezone

from scripts.run_open_interest_taker_flow import (
    OI_CONTRACTION_THRESHOLD,
    PRICE_MOVE_THRESHOLD,
    TAKER_LONG_THRESHOLD,
    _signal,
)
from scripts.run_momentum_volatility_research import Bar

UTC = timezone.utc


def bar(timestamp, close):
    return Bar(timestamp=timestamp, open=close, high=close, low=close, close=close, volume=1.0)


def test_up_move_with_oi_contraction_and_buyers_continues():
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    bars = [bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    bars[-1] = bar(start + timedelta(hours=6), 100.6)
    metrics = {
        start + timedelta(hours=i): {"oi": 100.0, "taker_ratio": TAKER_LONG_THRESHOLD}
        for i in range(7)
    }
    metrics[start + timedelta(hours=6)]["oi"] = 98.9
    signal = _signal(bars, metrics, 6)
    assert signal is not None
    assert signal[0] == 1
    assert signal[1] >= PRICE_MOVE_THRESHOLD
    assert signal[2] <= -OI_CONTRACTION_THRESHOLD


def test_down_move_with_oi_contraction_and_sellers_continues():
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    bars = [bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    bars[-1] = bar(start + timedelta(hours=6), 99.4)
    metrics = {
        start + timedelta(hours=i): {"oi": 100.0, "taker_ratio": 1.0 / TAKER_LONG_THRESHOLD}
        for i in range(7)
    }
    metrics[start + timedelta(hours=6)]["oi"] = 98.9
    signal = _signal(bars, metrics, 6)
    assert signal is not None
    assert signal[0] == -1


def test_missing_or_small_input_is_not_signal():
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    bars = [bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    metrics = {
        start + timedelta(hours=i): {"oi": 100.0, "taker_ratio": 1.0}
        for i in range(6)
    }
    assert _signal(bars, metrics, 6) is None
