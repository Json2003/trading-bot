"""Cross-exchange arbitrage strategy for the lightweight core protocol."""

from __future__ import annotations

from typing import Dict, List

from ..strategy import Bar, OrderIntent, Strategy


class CrossExArb(Strategy):
    """Look for price dislocations between two venues for the same symbol."""

    name = "arbitrage"

    def __init__(
        self,
        symbol: str,
        primary_exchange: str,
        hedge_exchange: str,
        min_edge_bps: float = 10.0,
        qty: float = 1.0,
    ) -> None:
        if min_edge_bps < 0:
            raise ValueError("min_edge_bps cannot be negative")
        if qty <= 0:
            raise ValueError("qty must be positive")

        self.symbols = [f"{primary_exchange}:{symbol}", f"{hedge_exchange}:{symbol}"]
        self._base_symbol = symbol
        self._primary = self.symbols[0]
        self._hedge = self.symbols[1]
        self._threshold = min_edge_bps / 1e4
        self._qty = qty

    def on_bar(self, bars: Dict[str, Bar]) -> List[OrderIntent]:
        primary = bars[self._primary]
        hedge = bars[self._hedge]

        if primary.close <= 0:
            return []

        spread = hedge.close - primary.close
        edge = spread / primary.close

        intents: List[OrderIntent] = []
        if edge > self._threshold:
            intents.append(
                OrderIntent(
                    idemp_key=f"arb-b-{primary.ts}",
                    symbol=self._primary,
                    side="buy",
                    qty=self._qty,
                    type="market",
                )
            )
            intents.append(
                OrderIntent(
                    idemp_key=f"arb-s-{primary.ts}",
                    symbol=self._hedge,
                    side="sell",
                    qty=self._qty,
                    type="market",
                )
            )
        elif edge < -self._threshold:
            intents.append(
                OrderIntent(
                    idemp_key=f"arb-s-{primary.ts}",
                    symbol=self._primary,
                    side="sell",
                    qty=self._qty,
                    type="market",
                )
            )
            intents.append(
                OrderIntent(
                    idemp_key=f"arb-b-{primary.ts}",
                    symbol=self._hedge,
                    side="buy",
                    qty=self._qty,
                    type="market",
                )
            )
        return intents

    def on_fill(self, fill: Dict[str, object] | None) -> None:  # pragma: no cover - hook for integration
        """Arbitrage strategy does not update state based on fills."""

    def risk_state(self) -> Dict[str, float]:
        return {
            "symbol": self._base_symbol,
            "primary": self._primary,
            "hedge": self._hedge,
            "threshold_bps": self._threshold * 1e4,
            "qty": self._qty,
        }


__all__ = ["CrossExArb"]
