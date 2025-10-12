"""Tests for the cross-exchange arbitrage strategy."""

from tradingbot_core.strategy import Bar
from tradingbot_core.strategies import CrossExchangeArbitrage


def make_bar(price: float) -> Bar:
    return Bar(ts=1, open=price, high=price, low=price, close=price, volume=1)


def test_no_trade_when_edge_not_met():
    strategy = CrossExchangeArbitrage("BTCUSDT", "exA", "exB", min_edge_bps=20, qty=1)
    bars = {
        "exA:BTCUSDT": make_bar(100),
        "exB:BTCUSDT": make_bar(100.1),
    }

    intents = strategy.on_bar(bars)

    assert intents == []


def test_buy_primary_sell_hedge_when_primary_cheaper():
    strategy = CrossExchangeArbitrage("BTCUSDT", "exA", "exB", min_edge_bps=10, qty=2)
    bars = {
        "exA:BTCUSDT": make_bar(100),
        "exB:BTCUSDT": make_bar(101.5),
    }

    intents = strategy.on_bar(bars)

    assert len(intents) == 2
    buy, sell = intents
    assert buy.side == "buy"
    assert buy.symbol == "exA:BTCUSDT"
    assert buy.qty == 2
    assert sell.side == "sell"
    assert sell.symbol == "exB:BTCUSDT"
    assert sell.qty == 2


def test_buy_hedge_sell_primary_when_hedge_cheaper():
    strategy = CrossExchangeArbitrage("BTCUSDT", "exA", "exB", min_edge_bps=10, qty=3)
    bars = {
        "exA:BTCUSDT": make_bar(101.5),
        "exB:BTCUSDT": make_bar(100),
    }

    intents = strategy.on_bar(bars)

    assert len(intents) == 2
    buy, sell = intents
    assert buy.side == "buy"
    assert buy.symbol == "exB:BTCUSDT"
    assert buy.qty == 3
    assert sell.side == "sell"
    assert sell.symbol == "exA:BTCUSDT"
    assert sell.qty == 3
