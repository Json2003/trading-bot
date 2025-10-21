"""Dollar-cost averaging strategy with optional martingale scaling."""

from __future__ import annotations

from typing import Iterable, List

from .base import Strategy, StrategyContext, StrategySignal


class DCAMartingaleStrategy(Strategy):
    """Progressively increases position size on drawdowns and exits on recovery."""

    def __init__(
        self,
        *,
        symbol: str,
        base_order_size: float,
        dca_step: float = 0.01,
        scale_factor: float = 1.4,
        max_layers: int = 4,
        take_profit: float = 0.01,
        venue: str | None = None,
        market_key: str | None = None,
    ) -> None:
        if dca_step <= 0:
            raise ValueError("dca_step must be positive")
        self.symbol = symbol
        self._base_size = base_order_size
        self._step = dca_step
        self._scale = scale_factor
        self._max_layers = max_layers
        self._take_profit = take_profit
        self._venue = venue
        self._market_key = market_key or (f"{venue}:{symbol}" if venue else symbol)
        self._position = 0.0
        self._avg_entry: float | None = None
        self._layers = 0

    def _sync_with_portfolio(self, context: StrategyContext, price: float) -> None:
        held_qty = 0.0
        held_avg: float | None = None
        for position in context.positions:
            if position.symbol == self.symbol:
                held_qty += position.quantity
                held_avg = position.average_price

        if abs(held_qty - self._position) <= 1e-9:
            return

        increasing = held_qty > self._position
        self._position = held_qty

        if held_qty <= 0:
            self._avg_entry = None
            self._layers = 0
            return

        if held_avg is not None:
            self._avg_entry = held_avg
        elif self._avg_entry is None:
            self._avg_entry = price

        if increasing:
            self._layers = min(self._layers + 1, self._max_layers)
        else:
            # Partial reductions reduce layer count but keep at least one while position remains.
            self._layers = max(1, self._layers - 1)

    def generate_signals(self, context: StrategyContext) -> Iterable[StrategySignal]:
        market = context.data(self._market_key)
        price = market.price
        self._sync_with_portfolio(context, price)

        signals: List[StrategySignal] = []

        if self._position <= 0:
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="buy",
                    quantity=self._base_size,
                    price=price,
                    venue=self._venue,
                    tags={
                        "type": "dca",
                        "layer": 1,
                        "market_key": self._market_key,
                    },
                )
            )
            return signals

        assert self._avg_entry is not None

        target_buy = self._avg_entry * (1.0 - self._step)
        if price <= target_buy and self._layers < self._max_layers:
            qty = self._base_size * (self._scale ** self._layers)
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="buy",
                    quantity=qty,
                    price=price,
                    venue=self._venue,
                    tags={
                        "type": "dca",
                        "layer": min(self._layers + 1, self._max_layers),
                        "market_key": self._market_key,
                    },
                )
            )
            return signals

        target_sell = self._avg_entry * (1.0 + self._take_profit)
        if price >= target_sell and self._position > 0:
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="sell",
                    quantity=self._position,
                    price=price,
                    venue=self._venue,
                    tags={
                        "type": "take_profit",
                        "layers": self._layers,
                        "market_key": self._market_key,
                    },
                )
            )

        return signals


__all__ = ["DCAMartingaleStrategy"]
