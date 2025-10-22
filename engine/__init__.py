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
from .beta_hedger import BetaHedgeCfg, BetaHedger
from .orchestrator import MultiStrategyOrchestrator, StrategyBinding
from .kill_switch import KillSwitchEvent, PortfolioKillSwitch
from .position_sizing import (
    ATRSizingConfig,
    PositionSizingResult,
    atr_position_size,
    atr_stop,
)
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
    "BetaHedgeCfg",
    "BetaHedger",
    "MultiStrategyOrchestrator",
    "StrategyBinding",
    "KillSwitchEvent",
    "PortfolioKillSwitch",
    "ATRSizingConfig",
    "PositionSizingResult",
    "atr_position_size",
    "atr_stop",
    "build_multi_strategy_orchestrator",
    "build_market_instruments",
    "build_strategy_allocations",
    "collect_market_instruments",
    "instantiate_strategy",
]
