"""Portfolio level accounting for multi-strategy execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping
from collections import defaultdict
import logging

from tradingbot_ibkr.execution.broker_base import BrokerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionView:
    """Lightweight snapshot of a held position."""

    symbol: str
    quantity: float
    average_price: float | None = None

    @property
    def market_value(self) -> float:
        if self.average_price is None:
            return 0.0
        return self.quantity * self.average_price


@dataclass(frozen=True)
class StrategyPnL:
    """Profit and loss snapshot."""

    realised: float
    unrealised: float


@dataclass
class StrategyAllocation:
    """Configuration describing how much capital a strategy can deploy."""

    name: str
    capital: float
    max_position_notional: float | None = None
    max_drawdown: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderFill:
    """Event representing a completed order fill."""

    symbol: str
    side: str
    quantity: float
    price: float
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True)
class StrategyState:
    """Current capital and exposures for a strategy."""

    allocation: StrategyAllocation
    cash: float
    positions: tuple[PositionView, ...]
    pnl: StrategyPnL


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time view of the whole book."""

    total_equity: float
    states: Mapping[str, StrategyState]

    def state_for(self, name: str) -> StrategyState:
        return self.states[name]


class Portfolio:
    """Track capital splits and aggregate realised/unrealised PnL."""

    def __init__(
        self,
        allocations: Iterable[StrategyAllocation],
        *,
        broker: BrokerBase,
        base_currency: str = "USD",
        log: logging.Logger | None = None,
    ) -> None:
        allocations = list(allocations)
        if not allocations:
            raise ValueError("At least one strategy allocation must be configured")
        self._allocations = {alloc.name: alloc for alloc in allocations}
        self._cash_balances = {alloc.name: alloc.capital for alloc in allocations}
        self._realised_pnl = defaultdict(float)
        self._broker = broker
        self._base_currency = base_currency
        self._log = log or logger

    def allocation_for(self, name: str) -> StrategyAllocation:
        return self._allocations[name]

    def apply_fills(self, strategy: str, fills: Iterable[OrderFill]) -> None:
        cash = self._cash_balances[strategy]
        for fill in fills:
            notional = fill.notional
            if fill.side.lower() == "buy":
                cash -= notional + fill.fee
            else:
                cash += notional - fill.fee
                self._realised_pnl[strategy] += notional - fill.fee
        self._cash_balances[strategy] = cash

    def snapshot(self, mark_prices: Mapping[str, float] | None = None) -> PortfolioSnapshot:
        positions = list(self._broker.list_positions())

        states: dict[str, StrategyState] = {}
        total_equity = 0.0
        for name, allocation in self._allocations.items():
            cash = self._cash_balances[name]
            held_positions: list[PositionView] = []
            unrealised = 0.0
            for position in positions:
                mark = None
                if mark_prices and position.symbol in mark_prices:
                    mark = mark_prices[position.symbol]
                elif position.average_price is not None:
                    mark = position.average_price
                if mark is None:
                    continue
                position_view = PositionView(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    average_price=mark,
                )
                held_positions.append(position_view)
                unrealised += (mark - (position.average_price or mark)) * position.quantity

            pnl = StrategyPnL(realised=self._realised_pnl[name], unrealised=unrealised)
            equity = cash + sum(p.market_value for p in held_positions) + pnl.realised + pnl.unrealised
            total_equity += equity
            states[name] = StrategyState(
                allocation=allocation,
                cash=cash,
                positions=tuple(held_positions),
                pnl=pnl,
            )
        return PortfolioSnapshot(total_equity=total_equity, states=states)

    def available_notional(self, strategy: str) -> float:
        allocation = self._allocations[strategy]
        limit = allocation.max_position_notional or allocation.capital
        consumed = allocation.capital - self._cash_balances[strategy]
        return max(limit - consumed, 0.0)


__all__ = [
    "OrderFill",
    "Portfolio",
    "PortfolioSnapshot",
    "PositionView",
    "StrategyAllocation",
    "StrategyPnL",
]
