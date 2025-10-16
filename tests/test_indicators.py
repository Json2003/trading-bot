from __future__ import annotations

import math

import pytest

from backtest.indicators import ATR, true_range


def test_true_range_matches_manual_calculation():
    assert math.isclose(true_range(100.0, 110.0, 90.0), 20.0)
    assert math.isclose(true_range(100.0, 101.0, 99.5), 1.5)


def test_atr_initial_window_uses_simple_average():
    atr = ATR(window=3)
    bars = [
        (0, 100.0, 110.0, 90.0, 105.0, 0),
        (1, 105.0, 115.0, 95.0, 110.0, 0),
        (2, 110.0, 118.0, 108.0, 112.0, 0),
    ]

    values = [atr.update(bar) for bar in bars]

    assert math.isclose(values[0], 20.0)
    assert math.isclose(values[1], 20.0)
    assert math.isclose(values[2], (20.0 + 20.0 + 10.0) / 3.0)


def test_atr_wilder_smoothing_after_window():
    atr = ATR(window=3)
    bars = [
        (0, 100.0, 110.0, 90.0, 105.0, 0),
        (1, 105.0, 115.0, 95.0, 110.0, 0),
        (2, 110.0, 118.0, 108.0, 112.0, 0),
        (3, 112.0, 120.0, 111.0, 119.0, 0),
    ]

    last_value = None
    for bar in bars:
        last_value = atr.update(bar)

    assert last_value is not None
    # ((prev_atr * (n-1)) + tr) / n
    expected = ((20.0 + 20.0 + 10.0) / 3.0 * 2.0 + 9.0) / 3.0
    assert math.isclose(last_value, expected)


def test_atr_accepts_object_like_bars():
    class Bar:
        def __init__(self, high: float, low: float, close: float) -> None:
            self.high = high
            self.low = low
            self.close = close

    atr = ATR(window=2)
    atr.prev_close = 100.0
    bar = Bar(103.0, 99.0, 101.0)
    value = atr.update(bar)
    assert math.isclose(value, true_range(100.0, 103.0, 99.0))


def test_atr_rejects_invalid_window():
    with pytest.raises(ValueError):
        ATR(window=0)

