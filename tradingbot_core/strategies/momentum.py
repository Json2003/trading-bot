"""Momentum strategy based on fast/slow exponential moving averages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..strategy import Bar, OrderIntent, Strategy


@dataclass(slots=True)
class _EMA:
    """Lightweight EMA accumulator."""

    window: int
    value: Optional[float] = None

    def update(self, price: float) -> float:
        alpha = 2.0 / (self.window + 1.0)
        if self.value is None:
            self.value = price
        else:
            self.value = alpha * price + (1.0 - alpha) * self.value
        return self.value


class MomentumEMA(Strategy):
    """Generate market orders when fast/slow EMA crosses with a threshold."""

    name = "momentum"

    def __init__(
        self,
        symbol: str,
        fast_window: int,
        slow_window: int,
        threshold_pct: float,
        order_qty: float = 1.0,
    ) -> None:
        if fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if slow_window <= 0:
            raise ValueError("slow_window must be positive")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if order_qty <= 0:
            raise ValueError("order_qty must be positive")
        if threshold_pct < 0:
            raise ValueError("threshold_pct cannot be negative")

        self.symbols = [symbol]
        self._fast = _EMA(fast_window)
        self._slow = _EMA(slow_window)
        self._threshold = threshold_pct / 100.0
        self._qty = order_qty
        self._bias = 0

    def on_bar(self, bars: Dict[str, Bar]) -> List[OrderIntent]:
        bar = bars[self.symbols[0]]
        price = bar.close

        fast = self._fast.update(price)
        slow = self._slow.update(price)
        edge = fast - slow
        band = self._threshold * price

        bias = 0
        if edge > band:
            bias = 1
        elif -edge > band:
            bias = -1

        if bias == self._bias or bias == 0:
            self._bias = bias
            return []

        self._bias = bias
        side = "buy" if bias > 0 else "sell"
        intent = OrderIntent(
            idemp_key=f"mom-{side[0]}-{bar.ts}",
            symbol=self.symbols[0],
            side=side,
            qty=self._qty,
            type="market",
        )
        return [intent]

    def on_fill(self, fill: Dict[str, object] | None) -> None:  # pragma: no cover - hook for integration
        """Momentum strategy does not react to fills in the lightweight example."""

    def risk_state(self) -> Dict[str, float]:
        return {
            "fast": float(self._fast.value or 0.0),
            "slow": float(self._slow.value or 0.0),
            "bias": float(self._bias),
            "threshold_pct": self._threshold * 100.0,
        }


__all__ = ["MomentumEMA"]
