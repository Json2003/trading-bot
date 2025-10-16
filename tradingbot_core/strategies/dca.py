"""Lightweight DCA martingale strategy built on the core protocol."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..strategy import Bar, OrderIntent, Strategy


class DCAMartingale(Strategy):
    """Implements a bounded martingale-style DCA accumulation strategy.

    The strategy anchors to the first observed close price.  Each subsequent
    drawdown of ``step_pct`` triggers an additional market buy order whose size
    doubles with every step, bounded by ``max_steps``.  The primary goal is to
    average into a position as price moves against the initial anchor.
    """

    name = "dca"

    def __init__(
        self,
        symbol: str,
        base_qty: float,
        step_pct: float = 2.0,
        max_steps: int = 4,
    ) -> None:
        if step_pct <= 0:
            raise ValueError("step_pct must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if base_qty <= 0:
            raise ValueError("base_qty must be positive")

        self.symbols = [symbol]
        self._base_qty = base_qty
        self._step = step_pct / 100.0
        self._max_steps = max_steps

        self._anchor: Optional[float] = None
        self._steps = 0

    def on_bar(self, bars: Dict[str, Bar]) -> List[OrderIntent]:
        symbol = self.symbols[0]
        price = bars[symbol].close

        if self._anchor is None:
            self._anchor = price
            return []

        drawdown = (self._anchor - price) / self._anchor
        intents: List[OrderIntent] = []

        while self._steps < self._max_steps and drawdown > self._step * (self._steps + 1):
            qty = self._base_qty * (2 ** self._steps)
            intents.append(
                OrderIntent(
                    idemp_key=f"dca-{self._steps}",
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    type="market",
                )
            )
            self._steps += 1

        return intents

    def on_fill(self, fill):  # type: ignore[override]
        """The strategy does not actively react to fills."""

    def risk_state(self) -> Dict[str, Optional[float]]:
        return {"steps": float(self._steps), "anchor": self._anchor}


__all__ = ["DCAMartingale"]
