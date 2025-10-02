"""Utilities for reconciling intended orders with broker state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .broker_base import Broker
from models import OrderRequest, OrderStatus

_TERMINAL_STATUSES = {
    "FILLED",
    "CANCELED",
    "CANCELLED",
    "REJECTED",
    "DONE",
    "EXPIRED",
}


@dataclass(slots=True)
class RiskLimits:
    """Configuration for broker level risk controls."""

    max_daily_loss_pct: float
    kill_switch_drawdown_pct: float
    max_position_risk_pct: float


class Reconciler:
    """Synchronise the intended orders with the broker state."""

    def __init__(self, broker: Broker, account_id: str, limits: RiskLimits, logger) -> None:
        self.broker = broker
        self.account_id = account_id
        self.limits = limits
        self.log = logger

    def submit_idempotent(self, order: OrderRequest) -> OrderStatus:
        """Submit ``order`` while avoiding duplicate broker entries."""

        status = self.broker.place_order(self.account_id, order)
        self.log.info(
            "submit_idempotent: status=%s id=%s",
            status.status,
            status.client_order_id,
        )
        return status

    @staticmethod
    def _order_key(order: OrderRequest | OrderStatus | Mapping[str, object]) -> str | None:
        """Return the preferred reconciliation key for an order-like payload."""

        # Support explicit idempotency hints provided either as attributes or via mappings.
        id_fields = ("idempotency_key", "idemp_key", "client_order_id")

        if isinstance(order, Mapping):
            for field in id_fields:
                value = order.get(field)
                if value:
                    return str(value)
            return None

        meta = getattr(order, "meta", None)
        if isinstance(meta, Mapping):
            for field in id_fields:
                value = meta.get(field)
                if value:
                    return str(value)

        for field in ("idemp_key", "client_id", "client_order_id"):
            value = getattr(order, field, None)
            if value:
                return str(value)

        return None

    def reconcile(self, intended_orders: Iterable[OrderRequest]) -> None:
        """Compare intended state vs broker state and heal any drift."""

        try:
            broker_orders = self.broker.list_orders(self.account_id)
        except NotImplementedError:
            broker_orders = ()

        open_now: dict[str, OrderStatus] = {}
        for broker_order in broker_orders:
            status = broker_order.status.upper()
            if status in _TERMINAL_STATUSES:
                continue

            key = self._order_key(broker_order)
            if key is not None:
                open_now[key] = broker_order

        try:
            positions = tuple(self.broker.get_positions(self.account_id))
        except NotImplementedError:
            positions = ()
        self.log.info("reconcile: open=%d positions=%s", len(open_now), positions)

        for order in intended_orders:
            key = self._order_key(order)
            if key is None or key not in open_now:
                self.submit_idempotent(order)

    def check_kill_switch(self, equity_curve: list[float]) -> bool:
        """Return ``True`` when the drawdown breaches the configured limit."""

        if not equity_curve:
            return False

        equity = equity_curve[-1]
        peak = max(equity_curve)
        drawdown_pct = 100.0 * (1 - equity / peak) if peak > 0 else 0.0
        if drawdown_pct >= self.limits.kill_switch_drawdown_pct:
            self.log.error("KILL SWITCH TRIGGERED: drawdown %.2f%%", drawdown_pct)
            return True
        return False


__all__ = ["RiskLimits", "Reconciler"]

