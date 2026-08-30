from datetime import datetime, timedelta, timezone

from scripts.run_open_interest_price_reversal import _signal
from scripts.run_momentum_volatility_research import Bar

UTC = timezone.utc

def _bar(timestamp, close):
    return Bar(timestamp=timestamp, open=100.0, high=100.0, low=100.0, close=close, volume=1.0)

def test_signal_source_is_unchanged_before_inversion():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    bars[-1] = _bar(start + timedelta(hours=6), 100.6)
    oi = {start + timedelta(hours=i): 100.0 for i in range(7)}
    oi[start + timedelta(hours=6)] = 101.1
    signal = _signal(bars, oi, 6)
    assert signal is not None
    original_side, price_move, oi_change = signal
    assert original_side == 1
    assert price_move >= 0.005
    assert oi_change >= 0.01

def test_missing_open_interest_is_unknown_not_zero():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=i), 100.0) for i in range(7)]
    oi = {start + timedelta(hours=i): 100.0 for i in range(6)}
    assert _signal(bars, oi, 6) is None
