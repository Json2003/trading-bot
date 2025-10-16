"""Execution agnostic orchestration primitives for the trading bot."""

from .ccxt_feed import CCXTFeed
from .datafeed import MarketData, MarketInstrument, UnifiedDataFeed
from .portfolio import (
    OrderFill,
    Portfolio,
    PortfolioSnapshot,
    PositionView,
    StrategyAllocation,
    StrategyPnL,
)
from .risk import RiskDecision, RiskManager, RiskViolation
from .orchestrator import MultiStrategyOrchestrator, StrategyBinding
from .builders import (
    build_multi_strategy_orchestrator,
    build_market_instruments,
    build_strategy_allocations,
    collect_market_instruments,
    instantiate_strategy,
)

__all__ = [
    "MarketData",
    "MarketInstrument",
    "UnifiedDataFeed",
    "CCXTFeed",
    "OrderFill",
    "Portfolio",
    "PortfolioSnapshot",
    "PositionView",
    "StrategyAllocation",
    "StrategyPnL",
    "RiskDecision",
    "RiskManager",
    "RiskViolation",
    "MultiStrategyOrchestrator",
    "StrategyBinding",
    "build_multi_strategy_orchestrator",
    "build_market_instruments",
    "build_strategy_allocations",
    "collect_market_instruments",
    "instantiate_strategy",
]
