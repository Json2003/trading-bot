"""Portfolio level accounting for multi-strategy execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, DefaultDict
from collections import defaultdict
import logging

from tradingbot_ibkr.execution.broker_base import BrokerBase
from tradingbot_ibkr.execution.broker_base import Position

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


@dataclass
class _TrackedPosition:
    """Internal helper tracking cost basis per strategy and symbol."""

    quantity: float = 0.0
    average_price: float = 0.0


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
        self._realised_pnl: DefaultDict[str, float] = defaultdict(float)
        self._position_tracker: dict[str, dict[str, _TrackedPosition]] = {
            alloc.name: {} for alloc in allocations
        }
        self._broker = broker
        self._base_currency = base_currency
        self._log = log or logger

    def allocation_for(self, name: str) -> StrategyAllocation:
        return self._allocations[name]

    def apply_fills(self, strategy: str, fills: Iterable[OrderFill]) -> None:
        cash = self._cash_balances[strategy]
        positions = self._position_tracker[strategy]
        realised = self._realised_pnl[strategy]
        for fill in fills:
            notional = fill.notional
            if fill.side.lower() == "buy":
                cash -= notional + fill.fee
                quantity_change = abs(fill.quantity)
            else:
                cash += notional - fill.fee
                quantity_change = -abs(fill.quantity)

            if quantity_change == 0:
                continue

            tracker = positions.get(fill.symbol)
            if tracker is None:
                tracker = _TrackedPosition()
                positions[fill.symbol] = tracker

            existing_qty = tracker.quantity
            avg_price = tracker.average_price

            if existing_qty == 0:
                tracker.quantity = quantity_change
                tracker.average_price = float(fill.price)
            elif (existing_qty > 0 and quantity_change > 0) or (
                existing_qty < 0 and quantity_change < 0
            ):
                total_qty = existing_qty + quantity_change
                tracker.quantity = total_qty
                tracker.average_price = (
                    (
                        abs(existing_qty) * avg_price
                        + abs(quantity_change) * float(fill.price)
                    )
                    / abs(total_qty)
                    if total_qty
                    else 0.0
                )
            else:
                closing_qty = min(abs(existing_qty), abs(quantity_change))
                realised += closing_qty * (float(fill.price) - avg_price) * (
                    1 if existing_qty > 0 else -1
                )
                remaining_qty = existing_qty + quantity_change
                tracker.quantity = remaining_qty
                if remaining_qty == 0:
                    tracker.average_price = 0.0
                    positions.pop(fill.symbol, None)
                elif (existing_qty > 0 and remaining_qty > 0) or (
                    existing_qty < 0 and remaining_qty < 0
                ):
                    tracker.average_price = avg_price
                else:
                    tracker.average_price = float(fill.price)

        self._realised_pnl[strategy] = realised
        self._cash_balances[strategy] = cash

    def snapshot(self, mark_prices: Mapping[str, float] | None = None) -> PortfolioSnapshot:
        broker_positions = {
            position.symbol: position
            for position in self._broker.list_positions()
        }
        mark_prices = dict(mark_prices or {})

        states: dict[str, StrategyState] = {}
        total_equity = 0.0
        for name, allocation in self._allocations.items():
            cash = self._cash_balances[name]
            held_positions: list[PositionView] = []
            unrealised = 0.0
            for symbol, tracker in self._position_tracker[name].items():
                quantity = tracker.quantity
                if quantity == 0:
                    continue
                mark = mark_prices.get(symbol)
                if mark is None:
                    broker_position: Position | None = broker_positions.get(symbol)
                    if broker_position and broker_position.average_price is not None:
                        mark = broker_position.average_price
                    else:
                        mark = tracker.average_price
                position_view = PositionView(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=mark,
                )
                held_positions.append(position_view)
                unrealised += (mark - tracker.average_price) * quantity

            pnl = StrategyPnL(realised=self._realised_pnl[name], unrealised=unrealised)
            equity = cash + sum(p.market_value for p in held_positions)
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
