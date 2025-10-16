"""Price grid based market making strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .base import Strategy, StrategyContext, StrategySignal


@dataclass
class GridLevelState:
    next_buy_index: int
    next_sell_index: int


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
        geometric: bool = False,
    ) -> None:
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")
        if levels < 2:
            raise ValueError("At least two levels are required")
        if geometric and (lower_bound <= 0 or upper_bound <= 0):
            raise ValueError("Geometric spacing requires positive bounds")
        self.symbol = symbol
        self._lower = lower_bound
        self._upper = upper_bound
        self._level_count = levels
        self._size = base_order_size
        self._venue = venue
        self._market_key = market_key or (f"{venue}:{symbol}" if venue else symbol)
        self._geometric = geometric
        self._price_levels = self._build_price_levels()
        self._state = GridLevelState(
            next_buy_index=min(1, len(self._price_levels) - 1),
            next_sell_index=max(len(self._price_levels) - 2, 0),
        )

    def _build_price_levels(self) -> List[float]:
        if self._geometric:
            ratio = (self._upper / self._lower) ** (1 / (self._level_count - 1))
            return [self._lower * (ratio**i) for i in range(self._level_count)]
        step = (self._upper - self._lower) / (self._level_count - 1)
        return [self._lower + i * step for i in range(self._level_count)]

    @property
    def price_levels(self) -> List[float]:
        """Return a copy of the configured price levels."""

        return list(self._price_levels)

    def generate_signals(self, context: StrategyContext) -> Iterable[StrategySignal]:
        market = context.data(self._market_key)
        price = market.price
        signals: List[StrategySignal] = []
        levels = self._price_levels

        while self._state.next_buy_index >= 0 and price <= levels[self._state.next_buy_index]:
            level = levels[self._state.next_buy_index]
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="buy",
                    quantity=self._size,
                    price=level,
                    venue=self._venue,
                    tags={
                        "type": "grid",
                        "level": level,
                        "spacing": "geometric" if self._geometric else "arithmetic",
                    },
                )
            )
            self._state.next_buy_index -= 1

        while (
            self._state.next_sell_index < len(levels)
            and price >= levels[self._state.next_sell_index]
        ):
            level = levels[self._state.next_sell_index]
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="sell",
                    quantity=self._size,
                    price=level,
                    venue=self._venue,
                    tags={
                        "type": "grid",
                        "level": level,
                        "spacing": "geometric" if self._geometric else "arithmetic",
                    },
                )
            )
            self._state.next_sell_index += 1

        return signals


__all__ = ["GridTradingStrategy"]
