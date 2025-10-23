"""Lightweight momentum strategy using dual exponential moving averages.

This module provides a compact strategy implementation that showcases how to
work with the :mod:`tradingbot_core.strategy` protocol.  The strategy keeps
track of a fast and a slow EMA and generates market orders whenever the fast
line crosses the slow one.  A very small ATR approximation is also maintained
so callers can translate the state into protective stops if desired.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .strategy import Bar, OrderIntent, Strategy


@dataclass(slots=True)
class _EMAState:
    """Helper structure encapsulating EMA calculation state."""

    span: int
    alpha: float
    window: Deque[float]
    value: Optional[float] = None

    def update(self, price: float) -> float:
        self.window.append(price)
        if self.value is None:
            self.value = price
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * price
        return self.value


class MomentumEMA(Strategy):
    """Dual EMA momentum strategy operating on :class:`Bar` inputs."""

    name = "momentum-ema"

    def __init__(
        self,
        symbol: str,
        fast: int = 12,
        slow: int = 26,
        qty: float = 0.1,
        atr_stop_mult: float = 2.0,
    ) -> None:
        if fast >= slow:
            raise ValueError("fast window must be strictly smaller than slow window")

        self.symbols = [symbol]
        self.fast = fast
        self.slow = slow
        self.qty = qty
        self.atr_mult = atr_stop_mult

        self._fast = _EMAState(fast, 2.0 / (fast + 1.0), deque(maxlen=fast))
        self._slow = _EMAState(slow, 2.0 / (slow + 1.0), deque(maxlen=slow))
        self._true_range = deque(maxlen=14)
        self._position = 0

    # -- internal helpers -------------------------------------------------
    def _update_atr(self, bar: Bar) -> float:
        self._true_range.append(bar.high - bar.low)
        if not self._true_range:
            return 0.0
        return sum(self._true_range) / len(self._true_range)

    def _atr(self) -> float:
        if not self._true_range:
            return 0.0
        return sum(self._true_range) / len(self._true_range)

    # -- Strategy protocol implementation --------------------------------
    def on_bar(self, bars: Dict[str, Bar]) -> List[OrderIntent]:
        bar = bars[self.symbols[0]]

        fast = self._fast.update(bar.close)
        slow = self._slow.update(bar.close)
        atr = self._update_atr(bar)

        # Wait until the slow window is "warmed up" to reduce noise.
        if len(self._slow.window) < self.slow:
            return []

        intents: List[OrderIntent] = []
        if fast > slow and self._position <= 0:
            intents.append(
                OrderIntent(
                    idemp_key=f"{self.name}-buy-{bar.ts}",
                    symbol=self.symbols[0],
                    side="buy",
                    qty=self.qty,
                    type="market",
                    meta={"atr": atr, "atr_mult": self.atr_mult},
                )
            )
            self._position = 1
        elif fast < slow and self._position >= 0:
            intents.append(
                OrderIntent(
                    idemp_key=f"{self.name}-sell-{bar.ts}",
                    symbol=self.symbols[0],
                    side="sell",
                    qty=self.qty,
                    type="market",
                    meta={"atr": atr, "atr_mult": self.atr_mult},
                )
            )
            self._position = -1

        return intents

    def on_fill(self, fill: Dict[str, object]) -> None:  # pragma: no cover - passthrough
        """Accept fills to satisfy the Strategy protocol."""

    def risk_state(self) -> Dict[str, float]:
        return {
            "position": float(self._position),
            "fast_ema": self._fast.value or 0.0,
            "slow_ema": self._slow.value or 0.0,
            "atr": self._atr(),
        }


__all__ = ["MomentumEMA"]

