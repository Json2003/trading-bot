from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from backtest.strategies.grid import generate_signals
from engine.datafeed import MarketData
from engine.portfolio import StrategyAllocation, StrategyPnL
from strategies.base import StrategyContext
from strategies.grid import GridTradingStrategy


def test_generate_signals_handles_basic_grid_distribution() -> None:
    df = pd.DataFrame({"close": [90, 100, 110]})

    out = generate_signals(df, levels=5, range_pct=0.1)

    assert out["signals"].to_list() == [1, 0, -1]


def test_generate_signals_accepts_capitalised_price_column() -> None:
    df = pd.DataFrame({"Close": [100, 99, 101]})

    out = generate_signals(df, levels=4, range_pct=0.02)

    assert set(out["signals"].to_list()).issubset({-1, 0, 1})


def make_context(price: float) -> StrategyContext:
    timestamp = datetime.now(timezone.utc)
    market = MarketData(
        venue="binance",
        symbol="BTC/USDT",
        timestamp=timestamp,
        price=price,
    )
    allocation = StrategyAllocation(name="grid", capital=100000.0)
    pnl = StrategyPnL(realised=0.0, unrealised=0.0)
    return StrategyContext(
        strategy="grid",
        timestamp=timestamp,
        market_data={market.key: market},
        allocation=allocation,
        cash=allocation.capital,
        positions=(),
        pnl=pnl,
    )


def test_geometric_price_levels_match_ratio() -> None:
    strategy = GridTradingStrategy(
        symbol="BTC/USDT",
        lower_bound=50000,
        upper_bound=70000,
        levels=15,
        base_order_size=0.005,
        venue="binance",
        geometric=True,
    )

    lower = 50000
    upper = 70000
    levels = 15
    ratio = (upper / lower) ** (1 / (levels - 1))
    expected = [lower * ratio**i for i in range(levels)]
    actual_levels = strategy.price_levels
    assert len(actual_levels) == len(expected)
    for actual, target in zip(actual_levels, expected, strict=True):
        assert math.isclose(actual, target, rel_tol=1e-9)


def test_geometric_strategy_generates_expected_sells() -> None:
    strategy = GridTradingStrategy(
        symbol="BTC/USDT",
        lower_bound=50000,
        upper_bound=70000,
        levels=15,
        base_order_size=0.005,
        venue="binance",
        geometric=True,
    )

    context = make_context(71000)
    signals = list(strategy.generate_signals(context))

    assert {signal.side for signal in signals} == {"sell"}
    prices = sorted(signal.price for signal in signals)
    levels = strategy.price_levels
    assert math.isclose(prices[0], levels[-2], rel_tol=1e-9)
    assert math.isclose(prices[1], levels[-1], rel_tol=1e-9)
    assert all(signal.quantity == 0.005 for signal in signals)


def test_geometric_strategy_generates_expected_buys() -> None:
    strategy = GridTradingStrategy(
        symbol="BTC/USDT",
        lower_bound=50000,
        upper_bound=70000,
        levels=15,
        base_order_size=0.005,
        venue="binance",
        geometric=True,
    )

    context = make_context(48000)
    signals = list(strategy.generate_signals(context))

    assert {signal.side for signal in signals} == {"buy"}
    prices = sorted(signal.price for signal in signals)
    levels = strategy.price_levels
    assert math.isclose(prices[0], levels[0], rel_tol=1e-9)
    assert math.isclose(prices[1], levels[1], rel_tol=1e-9)
    assert all(signal.quantity == 0.005 for signal in signals)
