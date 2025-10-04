"""Strategy interface and shared helper data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Protocol, Sequence

if False:  # pragma: no cover - type checking helper
    from engine.datafeed import MarketData
    from engine.portfolio import PositionView, StrategyAllocation, StrategyPnL


@dataclass(frozen=True)
class StrategySignal:
    """Signal instructing the execution layer to place an order."""

    strategy: str
    symbol: str
    side: str
    quantity: float
    price: float
    venue: str | None = None
    tags: Mapping[str, object] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True)
class StrategyContext:
    """Context passed to strategies on each evaluation cycle."""

    strategy: str
    timestamp: datetime
    market_data: Mapping[str, "MarketData"]
    allocation: "StrategyAllocation"
    cash: float
    positions: Sequence["PositionView"]
    pnl: "StrategyPnL"

    def data(self, key: str) -> "MarketData":
        return self.market_data[key]


class Strategy(Protocol):
    """Protocol implemented by trading strategies."""

    def generate_signals(self, context: StrategyContext) -> Iterable[StrategySignal]:
        ...


__all__ = ["Strategy", "StrategyContext", "StrategySignal"]
