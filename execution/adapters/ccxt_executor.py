"""Execution adapter that converts strategy signals into canonical broker orders."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from engine.portfolio import OrderFill
from strategies import StrategySignal
from tradingbot_ibkr.execution.broker_base import BrokerBase, Order, OrderStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from engine.orchestrator import OrderExecutor

logger = logging.getLogger(__name__)


@dataclass
class CCXTSignalExecutor:
    """Submit :class:`StrategySignal` objects through the shared broker contract."""

    broker: BrokerBase
    fees_bps: float = 0.0
    slippage_bps: float = 0.0
    log: logging.Logger | None = None
    _signal_keys: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._log = self.log or logger
        self._fee_rate = (self.fees_bps + self.slippage_bps) / 10_000.0

    def execute(self, signal: StrategySignal) -> OrderFill | None:
        order = self._to_order(signal)
        try:
            submitted = self.broker.submit_order(order)
        except Exception as exc:  # pragma: no cover - broker specific error
            self._log.warning("Failed to submit order for %s: %s", signal.symbol, exc)
            return None

        filled_qty = 0.0
        if isinstance(submitted, Order):
            submitted = self._auto_fill(submitted)
            filled_qty = float(submitted.filled_quantity or 0.0)
            if submitted.status == "filled" and filled_qty <= 0:
                filled_qty = float(submitted.quantity)
        elif isinstance(submitted, OrderStatus):
            filled_qty = float(submitted.filled_qty or 0.0)
        else:  # pragma: no cover - compatibility guard for third-party brokers
            filled_qty = float(getattr(submitted, "filled_quantity", 0.0) or 0.0)

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

    def _to_order(self, signal: StrategySignal) -> Order:
        tags: Mapping[str, Any] = signal.tags if isinstance(signal.tags, Mapping) else {}
        provided_key = tags.get("idempotency_key") or tags.get("signal_id")
        if provided_key:
            key = str(provided_key)
        else:
            key = self._signal_keys.setdefault(id(signal), uuid.uuid4().hex)

        return Order(
            id=key,
            client_order_id=key,
            idemp_key=key,
            symbol=signal.symbol,
            side=signal.side,
            quantity=float(signal.quantity),
            price=float(signal.price),
            order_type="market",
            metadata={
                "strategy": signal.strategy,
                "venue": signal.venue,
                "tags": dict(tags),
            },
        )

    def _auto_fill(self, order: Order) -> Order:
        """Mark the order filled when the selected paper broker supports it."""

        fill_order = getattr(self.broker, "fill_order", None)
        if callable(fill_order):
            try:
                return fill_order(order.id)
            except Exception:  # pragma: no cover - optional broker behaviour
                return order
        return order


__all__ = ["CCXTSignalExecutor"]
