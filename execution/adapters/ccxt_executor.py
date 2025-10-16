"""Execution adapter that converts strategy signals into broker orders."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

from engine.portfolio import OrderFill
from strategies import StrategySignal
from tradingbot_ibkr.execution.broker_base import BrokerBase, Order

if TYPE_CHECKING:  # pragma: no cover - typing only
    from engine.orchestrator import OrderExecutor

logger = logging.getLogger(__name__)


@dataclass
class CCXTSignalExecutor:
    """Submit :class:`StrategySignal` objects via a :class:`BrokerBase` instance."""

    broker: BrokerBase
    fees_bps: float = 0.0
    slippage_bps: float = 0.0
    log: logging.Logger | None = None

    def __post_init__(self) -> None:
        self._log = self.log or logger
        self._fee_rate = (self.fees_bps + self.slippage_bps) / 10_000.0

    def execute(self, signal: StrategySignal) -> OrderFill | None:
        try:
            order = self.broker.submit_order(
                signal.symbol,
                signal.side,
                signal.quantity,
                price=signal.price,
            )
        except Exception as exc:  # pragma: no cover - broker specific error
            self._log.warning("Failed to submit order for %s: %s", signal.symbol, exc)
            return None

        order = self._auto_fill(order)
        filled_qty = order.filled_quantity or order.quantity
        if filled_qty <= 0:
            return None

        fee = abs(signal.price * filled_qty) * self._fee_rate
        return OrderFill(
            symbol=signal.symbol,
            side=signal.side,
            quantity=filled_qty,
            price=float(signal.price),
            fee=fee,
        )

    def _auto_fill(self, order: Order) -> Order:
        """Attempt to mark the order as filled for brokers that support it."""

        if hasattr(self.broker, "fill_order"):
            try:
                return getattr(self.broker, "fill_order")(order.id)
            except Exception:  # pragma: no cover - optional behaviour
                return order
        return order


__all__ = ["CCXTSignalExecutor"]
