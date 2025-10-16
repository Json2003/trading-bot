"""Factory helpers for assembling the multi-strategy trading engine."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import logging

from tradingbot_ibkr.execution.broker_base import BrokerBase

from .datafeed import MarketInstrument, UnifiedDataFeed
from .orchestrator import MultiStrategyOrchestrator, OrderExecutor, StrategyBinding
from .portfolio import Portfolio, StrategyAllocation
from .risk import RiskManager
from config.portfolio_loader import PortfolioConfig
from strategies import (
    CrossExchangeArbitrageStrategy,
    DCAMartingaleStrategy,
    GridTradingStrategy,
    MomentumEMAStrategy,
    Strategy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_strategy_allocations",
    "instantiate_strategy",
    "collect_market_instruments",
    "build_market_instruments",
    "build_multi_strategy_orchestrator",
]


def build_market_instruments(
    strategy_params: Mapping[str, Mapping[str, Any]],
    *,
    default_timeframe: str,
) -> Sequence[MarketInstrument]:
    return collect_market_instruments(strategy_params, default_timeframe=default_timeframe)


def build_strategy_allocations(config: PortfolioConfig) -> list[StrategyAllocation]:
    """Convert portfolio configuration into :class:`StrategyAllocation` objects."""

    return [
        StrategyAllocation(
            name=strategy.name,
            capital=strategy.capital,
            max_position_notional=strategy.max_position_notional,
            max_drawdown=strategy.max_drawdown,
            metadata={"config": strategy.name},
        )
        for strategy in config.strategies
    ]


def instantiate_strategy(name: str, params: Mapping[str, Any]) -> Strategy:
    """Instantiate the strategy implementation for ``name`` using ``params``."""

    builder = _STRATEGY_BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"Unsupported strategy {name!r}")
    return builder(params)


def collect_market_instruments(
    strategy_params: Mapping[str, Mapping[str, Any]],
    *,
    default_timeframe: str | None = None,
) -> list[MarketInstrument]:
    """Return the unique set of :class:`MarketInstrument` objects needed for trading."""

    instruments: dict[str, MarketInstrument] = {}
    for params in strategy_params.values():
        for key, timeframe in _extract_market_keys(params, default_timeframe):
            venue, symbol = _split_market_key(key)
            alias = key
            if alias not in instruments:
                instruments[alias] = MarketInstrument(
                    venue=venue,
                    symbol=symbol,
                    timeframe=timeframe,
                    alias=alias,
                )
    return list(instruments.values())


def build_multi_strategy_orchestrator(
    *,
    portfolio_config: PortfolioConfig,
    strategy_params: Mapping[str, Mapping[str, Any]],
    clients: Mapping[str, Any],
    broker: BrokerBase,
    executor: OrderExecutor,
    default_timeframe: str = "1h",
    ohlcv_candles: int = 20,
    log: logging.Logger | None = None,
) -> tuple[
    MultiStrategyOrchestrator,
    Portfolio,
    RiskManager,
    UnifiedDataFeed,
    Sequence[StrategyBinding],
]:
    """Construct the orchestrator, portfolio and dependencies for the strategy set."""

    allocations = build_strategy_allocations(portfolio_config)
    allocation_map = {alloc.name: alloc for alloc in allocations}

    strategies = {
        name: instantiate_strategy(name, params) for name, params in strategy_params.items()
    }

    bindings: list[StrategyBinding] = []
    for cfg in portfolio_config.strategies:
        strategy = strategies.get(cfg.name)
        if strategy is None:
            raise KeyError(f"Missing strategy parameters for {cfg.name!r}")
        bindings.append(
            StrategyBinding(
                name=cfg.name,
                strategy=strategy,
                allocation=allocation_map[cfg.name],
            )
        )

    instruments = collect_market_instruments(
        strategy_params, default_timeframe=default_timeframe
    )

    data_feed = UnifiedDataFeed(
        clients=clients,
        instruments=instruments,
        ohlcv_candles=ohlcv_candles,
        default_timeframe=default_timeframe,
        log=log or logger,
    )

    portfolio = Portfolio(
        allocations,
        broker=broker,
        base_currency=portfolio_config.base_currency,
        log=log or logger,
    )

    risk_manager = RiskManager(
        allocation_map,
        portfolio_limits=portfolio_config.portfolio_limits,
        log=log or logger,
    )

    orchestrator = MultiStrategyOrchestrator(
        data_feed=data_feed,
        portfolio=portfolio,
        risk_manager=risk_manager,
        strategies=bindings,
        executor=executor,
        log=log or logger,
    )

    return orchestrator, portfolio, risk_manager, data_feed, bindings


def _split_market_key(key: str) -> tuple[str, str]:
    if ":" not in key:
        raise ValueError(
            "Market keys must be formatted as '<venue>:<symbol>' (e.g. binance:BTC/USDT)"
        )
    venue, symbol = key.split(":", 1)
    return venue, symbol


def _extract_market_keys(
    params: Mapping[str, Any], default_timeframe: str | None
) -> Sequence[tuple[str, str | None]]:
    keys: list[tuple[str, str | None]] = []
    if "market_key" in params:
        keys.append((str(params["market_key"]), params.get("timeframe") or default_timeframe))
    if "primary_market_key" in params:
        keys.append(
            (
                str(params["primary_market_key"]),
                params.get("primary_timeframe")
                or params.get("timeframe")
                or default_timeframe,
            )
        )
    if "hedge_market_key" in params:
        keys.append(
            (
                str(params["hedge_market_key"]),
                params.get("hedge_timeframe")
                or params.get("timeframe")
                or default_timeframe,
            )
        )
    return keys


def _build_grid(params: Mapping[str, Any]) -> Strategy:
    return GridTradingStrategy(
        symbol=str(params["symbol"]),
        lower_bound=float(params["lower_bound"]),
        upper_bound=float(params["upper_bound"]),
        levels=int(params["levels"]),
        base_order_size=float(params["base_order_size"]),
        venue=params.get("venue"),
        market_key=params.get("market_key"),
        geometric=bool(params.get("geometric", False)),
    )


def _build_momentum(params: Mapping[str, Any]) -> Strategy:
    return MomentumEMAStrategy(
        symbol=str(params["symbol"]),
        fast_window=int(params.get("fast_window", 12)),
        slow_window=int(params.get("slow_window", 26)),
        threshold=float(params.get("threshold", 0.0)),
        order_size=float(params.get("order_size", 1.0)),
        venue=params.get("venue"),
        market_key=params.get("market_key"),
    )


def _build_dca(params: Mapping[str, Any]) -> Strategy:
    return DCAMartingaleStrategy(
        symbol=str(params["symbol"]),
        base_order_size=float(params["base_order_size"]),
        dca_step=float(params.get("dca_step", 0.01)),
        scale_factor=float(params.get("scale_factor", 1.4)),
        max_layers=int(params.get("max_layers", 4)),
        take_profit=float(params.get("take_profit", 0.01)),
        venue=params.get("venue"),
        market_key=params.get("market_key"),
    )


def _build_arbitrage(params: Mapping[str, Any]) -> Strategy:
    return CrossExchangeArbitrageStrategy(
        primary_market_key=str(params["primary_market_key"]),
        hedge_market_key=str(params["hedge_market_key"]),
        trade_size=float(params["trade_size"]),
        min_edge=float(params.get("min_edge", 0.0)),
        fee_rate=float(params.get("fee_rate", 0.0)),
        symbol=params.get("symbol"),
    )


_STRATEGY_BUILDERS: Mapping[str, Any] = {
    "grid": _build_grid,
    "momentum": _build_momentum,
    "dca": _build_dca,
    "arbitrage": _build_arbitrage,
}
