from datetime import datetime, timedelta, timezone

import pytest

from scripts.run_open_interest_price_continuation import (
    OI_EXPANSION_THRESHOLD,
    PRICE_MOVE_THRESHOLD,
    _signal,
)
from scripts.run_momentum_volatility_research import Bar

UTC = timezone.utc


def _bar(timestamp, close, open_price=None):
    price = close if open_price is None else open_price
    return Bar(timestamp=timestamp, open=price, high=price, low=price, close=close, volume=1.0)


def test_price_move_with_oi_expansion_emits_same_direction():
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    bars[-1] = _bar(start + timedelta(hours=6), 100.6)
    oi = {start + timedelta(hours=i): 100.0 for i in range(7)}
    oi[start + timedelta(hours=6)] = 101.1
    signal = _signal(bars, oi, 6)
    assert signal is not None
    side, price_move, oi_change = signal
    assert side == 1
    assert price_move >= PRICE_MOVE_THRESHOLD
    assert oi_change >= OI_EXPANSION_THRESHOLD


def test_missing_open_interest_is_unknown_not_zero():
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    oi = {start + timedelta(hours=i): 100.0 for i in range(6)}
    assert _signal(bars, oi, 6) is None


def test_small_move_does_not_emit():
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    bars[-1] = _bar(start + timedelta(hours=6), 100.4)
    oi = {start + timedelta(hours=i): 100.0 for i in range(7)}
    oi[start + timedelta(hours=6)] = 101.1
    assert _signal(bars, oi, 6) is None
