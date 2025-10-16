"""Simple cross-exchange arbitrage strategy built on the core protocol."""

from __future__ import annotations

from typing import Dict, List

from ..strategy import Bar, OrderIntent, Strategy


class CrossExchangeArbitrage(Strategy):
    """Arbitrage strategy comparing prices across two venues."""

    name = "arbitrage"

    def __init__(
        self,
        symbol: str,
        primary_exchange: str,
        hedge_exchange: str,
        *,
        min_edge_bps: float = 15,
        qty: float = 0.01,
    ) -> None:
        self.symbols = [symbol]
        self._primary_exchange = primary_exchange
        self._hedge_exchange = hedge_exchange
        self._edge = min_edge_bps / 1e4
        self._qty = qty

    def on_bar(self, bars: Dict[str, Bar]) -> List[OrderIntent]:
        symbol = self.symbols[0]
        primary_key = f"{self._primary_exchange}:{symbol}"
        hedge_key = f"{self._hedge_exchange}:{symbol}"
        primary_price = bars[primary_key].close
        hedge_price = bars[hedge_key].close

        intents: List[OrderIntent] = []
        if primary_price * (1 + self._edge) < hedge_price:
            intents.extend(
                [
                    OrderIntent(
                        f"arb-b-{self._primary_exchange}",
                        primary_key,
                        "buy",
                        self._qty,
                        "market",
                    ),
                    OrderIntent(
                        f"arb-s-{self._hedge_exchange}",
                        hedge_key,
                        "sell",
                        self._qty,
                        "market",
                    ),
                ]
            )
        elif hedge_price * (1 + self._edge) < primary_price:
            intents.extend(
                [
                    OrderIntent(
                        f"arb-b-{self._hedge_exchange}",
                        hedge_key,
                        "buy",
                        self._qty,
                        "market",
                    ),
                    OrderIntent(
                        f"arb-s-{self._primary_exchange}",
                        primary_key,
                        "sell",
                        self._qty,
                        "market",
                    ),
                ]
            )
        return intents

    def on_fill(self, fill: Dict[str, object]) -> None:
        """Handle fills (no-op for the simple strategy)."""

    def risk_state(self) -> Dict[str, object]:
        """Return the empty risk state for compatibility with the protocol."""

        return {}


__all__ = ["CrossExchangeArbitrage"]
