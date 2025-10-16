from __future__ import annotations

import pytest

from config.portfolio_loader import load_portfolio_config, load_strategy_params
from engine.builders import (
    build_multi_strategy_orchestrator,
    build_strategy_allocations,
    collect_market_instruments,
)
from execution.adapters.ccxt_executor import CCXTSignalExecutor
from strategies.base import StrategySignal
from tradingbot_ibkr.execution.paper_broker import PaperBroker


class StubExchange:
    def __init__(self, tickers: dict[str, float], ohlcv: dict[str, list[list[float]]]):
        self.tickers = tickers
        self.ohlcv = ohlcv

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": self.tickers[symbol], "timestamp": 1_600_000_000_000}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        return self.ohlcv.get(symbol, [])


def test_load_portfolio_configuration():
    cfg = load_portfolio_config("config/portfolio.yaml")
    assert cfg.base_currency == "USD"
    assert cfg.total_capital == 100_000
    assert "max_symbol_notional" in cfg.portfolio_limits

    allocations = build_strategy_allocations(cfg)
    assert [a.name for a in allocations] == [s.name for s in cfg.strategies]
    grid_alloc = next(a for a in allocations if a.name == "grid")
    assert grid_alloc.capital == 25_000
    assert grid_alloc.metadata["config"] == "grid"


def test_collect_market_instruments_from_configs():
    cfg = load_portfolio_config("config/portfolio.yaml")
    params = {
        strat.name: load_strategy_params(strat.name, "config/strategy")
        for strat in cfg.strategies
    }
    instruments = collect_market_instruments(params, default_timeframe="1h")
    keys = {instrument.key() for instrument in instruments}
    assert keys == {"binance:BTC/USDT", "binance:ETH/USDT", "coinbase:BTC/USDT"}
    for instrument in instruments:
        assert instrument.alias == instrument.key()


def test_multi_strategy_orchestrator_cycle():
    cfg = load_portfolio_config("config/portfolio.yaml")
    params = {
        strat.name: load_strategy_params(strat.name, "config/strategy")
        for strat in cfg.strategies
    }

    binance = StubExchange(
        tickers={"BTC/USDT": 48_000.0, "ETH/USDT": 2_120.0},
        ohlcv={
            "ETH/USDT": [
                [0, 1_800, 1_850, 1_790, 1_800, 100],
                [1, 1_850, 1_880, 1_840, 1_860, 120],
                [2, 1_900, 1_940, 1_880, 1_920, 130],
                [3, 1_950, 1_980, 1_930, 1_970, 140],
                [4, 2_000, 2_050, 1_990, 2_020, 150],
            ],
            "BTC/USDT": [[0, 50_000, 51_000, 49_000, 50_500, 200]],
        },
    )
    coinbase = StubExchange(
        tickers={"BTC/USDT": 49_200.0},
        ohlcv={"BTC/USDT": [[0, 49_500, 49_800, 49_200, 49_500, 180]]},
    )

    clients = {"binance": binance, "coinbase": coinbase}

    broker = PaperBroker()
    executor = CCXTSignalExecutor(broker, fees_bps=0.0, slippage_bps=0.0)

    orchestrator, portfolio, _, _, _ = build_multi_strategy_orchestrator(
        portfolio_config=cfg,
        strategy_params=params,
        clients=clients,
        broker=broker,
        executor=executor,
        default_timeframe="1h",
        ohlcv_candles=5,
    )

    fills = orchestrator.run_cycle()
    assert fills, "Expected at least one fill from the strategy suite"

    # Ensure the broker registered executions
    positions = list(broker.list_positions())
    assert positions, "Paper broker should hold at least one position after execution"

    snapshot = portfolio.snapshot()
    assert snapshot.total_equity >= 100_000.0


def test_ccxt_signal_executor_fee_model():
    broker = PaperBroker()
    executor = CCXTSignalExecutor(broker, fees_bps=10.0, slippage_bps=5.0)
    signal = StrategySignal(
        strategy="test",
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        price=20_000.0,
    )
    fill = executor.execute(signal)
    assert fill is not None
    expected_fee = 0.1 * 20_000 * (0.0015)
    assert abs(fill.fee - expected_fee) < 1e-6
    positions = list(broker.list_positions())
    assert positions and positions[0].symbol == "BTC/USDT"
