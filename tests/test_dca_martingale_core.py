"""Unit tests for the lightweight DCA martingale strategy."""

import pytest

from tradingbot_core.strategy import Bar
from tradingbot_core.strategies import DCAMartingale


def make_bar(close: float) -> Bar:
    return Bar(ts=1, open=close, high=close, low=close, close=close, volume=1)


def test_dca_anchors_on_first_bar():
    strategy = DCAMartingale(symbol="BTCUSDT", base_qty=1.0, step_pct=5.0, max_steps=3)

    signals = strategy.on_bar({"BTCUSDT": make_bar(100.0)})

    assert signals == []
    state = strategy.risk_state()
    assert state["steps"] == 0.0
    assert state["anchor"] == 100.0


def test_dca_generates_progressive_orders_on_drawdown():
    strategy = DCAMartingale(symbol="BTCUSDT", base_qty=1.0, step_pct=5.0, max_steps=3)

    # Establish anchor at 100
    strategy.on_bar({"BTCUSDT": make_bar(100.0)})

    # 12% drawdown should trigger two layers (5% and 10%)
    signals = strategy.on_bar({"BTCUSDT": make_bar(88.0)})

    assert len(signals) == 2
    assert signals[0].qty == 1.0
    assert signals[0].idemp_key == "dca-0"
    assert signals[1].qty == 2.0
    assert signals[1].idemp_key == "dca-1"

    # Further drawdown beyond max_steps does not exceed cap
    signals = strategy.on_bar({"BTCUSDT": make_bar(50.0)})
    assert len(signals) == 1
    assert signals[0].qty == 4.0
    assert signals[0].idemp_key == "dca-2"

    # Additional drawdown still respects max_steps
    assert strategy.on_bar({"BTCUSDT": make_bar(1.0)}) == []


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        DCAMartingale(symbol="BTCUSDT", base_qty=0.0)

    with pytest.raises(ValueError):
        DCAMartingale(symbol="BTCUSDT", base_qty=1.0, step_pct=0.0)

    with pytest.raises(ValueError):
        DCAMartingale(symbol="BTCUSDT", base_qty=1.0, max_steps=0)
