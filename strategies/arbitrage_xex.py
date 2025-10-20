"""Cross-exchange arbitrage strategy operating on pre-positioned balances."""

from __future__ import annotations

from typing import Iterable, List

from .base import Strategy, StrategyContext, StrategySignal


class CrossExchangeArbitrageStrategy(Strategy):
    """Looks for dislocations between two venues and trades the spread."""

    def __init__(
        self,
        *,
        primary_market_key: str,
        hedge_market_key: str,
        trade_size: float,
        min_edge: float = 0.001,
        fee_rate: float = 0.0005,
        symbol: str | None = None,
    ) -> None:
        self._primary_key = primary_market_key
        self._hedge_key = hedge_market_key
        self._size = trade_size
        self._min_edge = min_edge
        self._fee = fee_rate
        self.symbol = symbol

    def generate_signals(self, context: StrategyContext) -> Iterable[StrategySignal]:
        primary = context.data(self._primary_key)
        hedge = context.data(self._hedge_key)
        symbol = self.symbol or primary.symbol

        spread = hedge.price - primary.price
        edge = spread / primary.price if primary.price else 0.0

        signals: List[StrategySignal] = []
        if edge > self._min_edge + 2 * self._fee:
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=symbol,
                    side="buy",
                    quantity=self._size,
                    price=primary.price,
                    venue=primary.venue,
                    tags={
                        "type": "xex",
                        "leg": "primary",
                        "edge": edge,
                        "market_key": self._primary_key,
                    },
                )
            )
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=symbol,
                    side="sell",
                    quantity=self._size,
                    price=hedge.price,
                    venue=hedge.venue,
                    tags={
                        "type": "xex",
                        "leg": "hedge",
                        "edge": edge,
                        "market_key": self._hedge_key,
                    },
                )
            )
        elif edge < -self._min_edge - 2 * self._fee:
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=symbol,
                    side="sell",
                    quantity=self._size,
                    price=primary.price,
                    venue=primary.venue,
                    tags={
                        "type": "xex",
                        "leg": "primary",
                        "edge": edge,
                        "market_key": self._primary_key,
                    },
                )
            )
            signals.append(
                StrategySignal(
                    strategy=context.strategy,
                    symbol=symbol,
                    side="buy",
                    quantity=self._size,
                    price=hedge.price,
                    venue=hedge.venue,
                    tags={
                        "type": "xex",
                        "leg": "hedge",
                        "edge": edge,
                        "market_key": self._hedge_key,
                    },
                )
            )
        return signals


__all__ = ["CrossExchangeArbitrageStrategy"]
