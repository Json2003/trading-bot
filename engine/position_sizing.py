"""Position sizing helpers used by engine-level orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from .datafeed import MarketData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionSizingResult:
    """Result returned by sizing helpers."""

    quantity: float
    notional: float
    risk_cash: float
    atr: float | None
    stop_distance: float | None

    @property
    def is_actionable(self) -> bool:
        """Return ``True`` when the sizing suggests a tradable position."""

        return self.quantity > 0 and self.notional > 0 and self.stop_distance not in (None, 0)


@dataclass(frozen=True)
class ATRSizingConfig:
    """Configuration driving :func:`atr_position_size`."""

    risk_fraction: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 2.0
    min_notional: float = 0.0
    min_quantity: float = 0.0
    max_notional: float | None = None
    max_leverage: float | None = None

    def __post_init__(self) -> None:
        if self.risk_fraction < 0:
            raise ValueError("risk_fraction must be non-negative")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")
        if self.min_notional < 0:
            raise ValueError("min_notional must be non-negative")
        if self.min_quantity < 0:
            raise ValueError("min_quantity must be non-negative")
        if self.max_notional is not None and self.max_notional <= 0:
            raise ValueError("max_notional must be positive when provided")
        if self.max_leverage is not None and self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive when provided")

def atr_position_size(
    equity: float,
    market: MarketData,
    *,
    config: ATRSizingConfig,
    price: float | None = None,
) -> PositionSizingResult:
    """Return a volatility-aware position size based on ATR.

    The helper computes the Average True Range for the instrument, translates
    the configured ``risk_fraction`` into risk capital and divides by the stop
    distance derived from ``atr * atr_multiplier``.  The resulting quantity is
    capped by ``max_notional`` and optional ``max_leverage`` constraints.
    """

    if equity <= 0:
        return PositionSizingResult(0.0, 0.0, 0.0, None, None)

    price = float(price if price is not None else market.price)
    if not price or price <= 0:
        logger.debug("Cannot size position: invalid price %s", price)
        return PositionSizingResult(0.0, 0.0, 0.0, None, None)

    risk_cash = max(equity * config.risk_fraction, 0.0)
    if risk_cash <= 0:
        logger.debug("Risk fraction %s produced zero risk cash", config.risk_fraction)
        return PositionSizingResult(0.0, 0.0, 0.0, None, None)

    atr_value = market.atr(config.atr_period)
    if atr_value is None or atr_value <= 0:
        logger.debug("ATR unavailable for %s", market.symbol)
        return PositionSizingResult(0.0, 0.0, risk_cash, atr_value, None)

    stop_distance = atr_value * config.atr_multiplier
    if stop_distance <= 0:
        return PositionSizingResult(0.0, 0.0, risk_cash, atr_value, stop_distance)

    quantity = risk_cash / stop_distance
    notional = quantity * price

    if config.max_leverage is not None:
        max_notional_from_lev = equity * config.max_leverage
        if notional > max_notional_from_lev:
            notional = max_notional_from_lev
            quantity = notional / price

    if config.max_notional is not None and notional > config.max_notional:
        notional = config.max_notional
        quantity = notional / price

    if notional < config.min_notional or quantity < config.min_quantity:
        return PositionSizingResult(0.0, 0.0, risk_cash, atr_value, stop_distance)

    return PositionSizingResult(quantity, notional, risk_cash, atr_value, stop_distance)


def atr_stop(price: float, stop_distance: float, side: str) -> float:
    """Return a stop-loss level derived from ATR sizing."""

    if stop_distance <= 0:
        return price
    if side.lower() == "buy":
        return price - stop_distance
    if side.lower() == "sell":
        return price + stop_distance
    return price


__all__ = [
    "ATRSizingConfig",
    "PositionSizingResult",
    "atr_position_size",
    "atr_stop",
]

