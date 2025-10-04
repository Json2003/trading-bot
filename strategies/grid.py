"""Price grid based market making strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .base import Strategy, StrategyContext, StrategySignal


@dataclass
class GridLevelState:
    next_buy: float
    next_sell: float


class GridTradingStrategy(Strategy):
    """Simple grid strategy placing layered buy/sell orders."""

    def __init__(
        self,
        *,
        symbol: str,
        lower_bound: float,
        upper_bound: float,
        levels: int,
        base_order_size: float,
        venue: str | None = None,
        market_key: str | None = None,
    ) -> None:
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")
        if levels < 2:
            raise ValueError("At least two levels are required")
        self.symbol = symbol
        self._lower = lower_bound
        self._upper = upper_bound
        self._levels = levels
        self._size = base_order_size
        self._venue = venue
        self._market_key = market_key or (f"{venue}:{symbol}" if venue else symbol)
        step = (upper_bound - lower_bound) / (levels - 1)
        self._state = GridLevelState(next_buy=lower_bound + step, next_sell=upper_bound - step)

    def generate_signals(self, context: StrategyContext) -> Iterable[StrategySignal]:
        market = context.data(self._market_key)
        price = market.price
        signals: List[StrategySignal] = []
        step = (self._upper - self._lower) / (self._levels - 1)

        while price <= self._state.next_buy and self._state.next_buy >= self._lower:
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="buy",
                    quantity=self._size,
                    price=self._state.next_buy,
                    venue=self._venue,
                    tags={"type": "grid", "level": self._state.next_buy},
                )
            )
            self._state = GridLevelState(
                next_buy=self._state.next_buy - step,
                next_sell=self._state.next_sell,
            )

        while price >= self._state.next_sell and self._state.next_sell <= self._upper:
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="sell",
                    quantity=self._size,
                    price=self._state.next_sell,
                    venue=self._venue,
                    tags={"type": "grid", "level": self._state.next_sell},
                )
            )
            self._state = GridLevelState(
                next_buy=self._state.next_buy,
                next_sell=self._state.next_sell + step,
            )

        return signals


__all__ = ["GridTradingStrategy"]
