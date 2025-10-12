"""Simple grid trading strategy using the lightweight strategy protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..strategy import Bar, OrderIntent, Strategy


@dataclass(slots=True, frozen=True)
class GridConfig:
    """Configuration parameters for :class:`GridStrategy`."""

    symbol: str
    lower: float
    upper: float
    levels: int
    quantity: float
    geometric: bool = True


class GridStrategy(Strategy):
    """Generate limit orders along a price grid to capture mean reversion."""

    name = "grid"

    def __init__(self, config: GridConfig) -> None:
        if config.levels < 2:
            raise ValueError("GridStrategy requires at least two levels")
        if config.lower <= 0 or config.upper <= 0:
            if config.geometric:
                raise ValueError("lower and upper must be positive when using geometric spacing")
        if config.lower >= config.upper:
            raise ValueError("lower must be strictly less than upper")
        if config.quantity <= 0:
            raise ValueError("quantity must be positive")

        self.symbol = config.symbol
        self.symbols: List[str] = [config.symbol]
        self._lower = config.lower
        self._upper = config.upper
        self._levels = config.levels
        self._qty = config.quantity
        self._geometric = config.geometric
        self._prices = self._build_price_levels()

    @staticmethod
    def _format_level(level: float) -> str:
        """Return a stable textual representation for grid levels.

        ``str(0.1)`` style formatting can leak floating point artefacts like
        ``"0.30000000000000004"`` which end up embedded in idempotency keys.
        Trimming the representation keeps the keys predictable while
        maintaining a trailing decimal digit so that legacy behaviour where
        levels were rendered as ``"100.0"`` is preserved.
        """

        text = format(level, ".10f").rstrip("0")
        if text.endswith("."):
            text += "0"
        return text

    def _build_price_levels(self) -> List[float]:
        if self._geometric:
            ratio = (self._upper / self._lower) ** (1 / (self._levels - 1))
            return [self._lower * (ratio ** i) for i in range(self._levels)]
        step = (self._upper - self._lower) / (self._levels - 1)
        return [self._lower + i * step for i in range(self._levels)]

    @property
    def prices(self) -> List[float]:
        """Return a copy of the current price grid."""

        return list(self._prices)

    def on_bar(self, bars: Dict[str, Bar]) -> List[OrderIntent]:
        bar = bars[self.symbol]
        price = bar.close
        intents: List[OrderIntent] = []
        for level in self._prices:
            if price < level:
                intents.append(
                    OrderIntent(
                        idemp_key=f"grid-b-{self._format_level(level)}",
                        symbol=self.symbol,
                        side="buy",
                        qty=self._qty,
                        type="limit",
                        limit_price=level,
                    )
                )
            elif price > level:
                intents.append(
                    OrderIntent(
                        idemp_key=f"grid-s-{self._format_level(level)}",
                        symbol=self.symbol,
                        side="sell",
                        qty=self._qty,
                        type="limit",
                        limit_price=level,
                    )
                )
        return intents

    def on_fill(self, fill: Dict[str, object] | None) -> None:  # pragma: no cover - hook for integration tests
        """Handle fills. Left intentionally empty for the lightweight example."""

    def risk_state(self) -> Dict[str, object]:
        """Return the static risk snapshot for the strategy."""

        return {"symbol": self.symbol, "levels": self._levels, "quantity": self._qty}


__all__ = ["GridConfig", "GridStrategy"]
