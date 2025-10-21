from __future__ import annotations

from datetime import datetime, timedelta, timezone

from engine.datafeed import MarketData, OHLCV
from engine.portfolio import StrategyAllocation, StrategyPnL
from strategies.base import StrategyContext
from strategies.momentum_ema import MomentumEMAStrategy


def _build_context(session: str, prices: list[float], high_low_spread: float = 20.0) -> StrategyContext:
    timestamp = datetime.now(timezone.utc)
    candles: list[OHLCV] = []
    start = timestamp - timedelta(minutes=len(prices))
    for idx, price in enumerate(prices):
        ts = start + timedelta(minutes=idx)
        candles.append(
            OHLCV(
                timestamp=ts,
                open=price,
                high=price + high_low_spread,
                low=price - high_low_spread,
                close=price,
                volume=50.0,
            )
        )

    market = MarketData(
        venue="binance",
        symbol="BTC/USDT",
        timestamp=timestamp,
        price=prices[-1],
        session=session,
        ohlcv=tuple(candles),
        metrics={"session": session},
    )
    allocation = StrategyAllocation(name="mom", capital=100000.0)
    pnl = StrategyPnL(realised=0.0, unrealised=0.0)

    return StrategyContext(
        strategy="mom",
        timestamp=timestamp,
        market_data={market.key: market},
        allocation=allocation,
        cash=allocation.capital,
        positions=(),
        pnl=pnl,
    )


def test_session_specific_thresholds_gate_signals() -> None:
    prices = [20000.0 + idx * 15 for idx in range(30)]
    context_asia = _build_context("asia", prices)
    context_us = _build_context("us", prices)

    strategy = MomentumEMAStrategy(
        symbol="BTC/USDT",
        venue="binance",
        fast_window=5,
        slow_window=21,
        threshold=0.001,
        session_thresholds={"asia": 0.02, "us": 0.001},
        atr_pct_threshold=None,
    )

    asia_signals = list(strategy.generate_signals(context_asia))
    assert not asia_signals

    us_signals = list(strategy.generate_signals(context_us))
    assert us_signals
    assert us_signals[0].side == "buy"


def test_volatility_filter_blocks_overheated_markets() -> None:
    prices = [20000.0 + idx * 10 for idx in range(40)]
    context_hot = _build_context("us", prices, high_low_spread=2000.0)
    context_normal = _build_context("us", prices, high_low_spread=25.0)

    noisy_strategy = MomentumEMAStrategy(
        symbol="BTC/USDT",
        venue="binance",
        fast_window=5,
        slow_window=21,
        threshold=0.001,
        atr_pct_threshold=0.01,
    )
    quiet_strategy = MomentumEMAStrategy(
        symbol="BTC/USDT",
        venue="binance",
        fast_window=5,
        slow_window=21,
        threshold=0.001,
        atr_pct_threshold=None,
    )

    assert not list(noisy_strategy.generate_signals(context_hot))
    assert list(quiet_strategy.generate_signals(context_normal))
