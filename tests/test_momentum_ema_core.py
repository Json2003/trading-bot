"""Tests for the lightweight momentum EMA strategy in ``tradingbot_core``."""

from tradingbot_core.momentum import MomentumEMA
from tradingbot_core.strategy import Bar


def _bar(ts: int, price: float) -> Bar:
    return Bar(ts=ts, open=price, high=price + 1, low=price - 1, close=price, volume=100)


def test_momentum_strategy_generates_cross_signals() -> None:
    strategy = MomentumEMA(symbol="BTCUSDT", fast=2, slow=4, qty=1.5)

    # Warm-up phase – no orders yet.
    for idx, price in enumerate([10, 11, 12]):
        intents = strategy.on_bar({"BTCUSDT": _bar(idx, price)})
        assert intents == []

    # Fast EMA now above slow EMA -> expect a buy signal.
    intents = strategy.on_bar({"BTCUSDT": _bar(3, 13)})
    assert len(intents) == 1
    first = intents[0]
    assert first.side == "buy"
    assert first.qty == 1.5
    assert first.symbol == "BTCUSDT"
    assert "atr" in first.meta

    # Continue in up-trend – no duplicate orders should be produced.
    assert strategy.on_bar({"BTCUSDT": _bar(4, 14)}) == []

    # Down-trend pushes fast EMA below slow EMA -> expect a sell signal.
    intents = strategy.on_bar({"BTCUSDT": _bar(5, 5)})
    assert len(intents) == 1
    second = intents[0]
    assert second.side == "sell"
    assert second.symbol == "BTCUSDT"


def test_risk_state_snapshot() -> None:
    strategy = MomentumEMA(symbol="ETHUSDT", fast=2, slow=5)
    strategy.on_bar({"ETHUSDT": _bar(0, 20)})
    state = strategy.risk_state()
    assert set(state) == {"position", "fast_ema", "slow_ema", "atr"}
    assert state["position"] in (-1.0, 0.0, 1.0)
