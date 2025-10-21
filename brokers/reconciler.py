"""Utilities for reconciling intended orders with broker state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

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

    def __init__(
        self,
        broker: Broker,
        limits: RiskLimits,
        logger,
        *,
        account_id: str | None = None,
    ) -> None:
        self.broker = broker
        self.account_id = account_id
        self.limits = limits
        self.log = logger

    # ---------------------------------------------------------------------
    # Broker helpers
    # ------------------------------------------------------------------
    def _place(self, order: OrderRequest | Mapping[str, Any] | object) -> OrderStatus:
        """Submit ``order`` via whichever broker API is available."""

        if hasattr(self.broker, "place_order"):
            account_id = "" if self.account_id is None else self.account_id
            return self.broker.place_order(account_id, order)  # type: ignore[arg-type]

        if hasattr(self.broker, "place"):
            # Legacy interface used by older broker implementations.
            return self.broker.place(order)  # type: ignore[attr-defined]

        raise AttributeError("Broker does not expose place_order/place")

    def _list_orders(self) -> Sequence[OrderStatus] | Iterable[OrderStatus]:
        if hasattr(self.broker, "list_orders"):
            account_id = "" if self.account_id is None else self.account_id
            return self.broker.list_orders(account_id)  # type: ignore[attr-defined]

        if hasattr(self.broker, "fetch_open_orders"):
            return self.broker.fetch_open_orders()  # type: ignore[attr-defined]

        raise NotImplementedError("Broker cannot provide open orders")

    def _list_positions(self) -> Sequence[object] | Iterable[object]:
        if hasattr(self.broker, "get_positions"):
            account_id = "" if self.account_id is None else self.account_id
            return self.broker.get_positions(account_id)  # type: ignore[attr-defined]

        if hasattr(self.broker, "fetch_positions"):
            return self.broker.fetch_positions()  # type: ignore[attr-defined]

        raise NotImplementedError("Broker cannot provide positions")

    # ------------------------------------------------------------------
    def submit_idempotent(self, order: OrderRequest | Mapping[str, Any] | object) -> OrderStatus:
        """Submit ``order`` while avoiding duplicate broker entries."""

        status = self._place(order)
        status_desc = None
        client_id = None

        if isinstance(status, Mapping):
            status_desc = status.get("status") or status.get("state")
            client_id = status.get("client_order_id") or status.get("client_id")
        else:
            status_desc = getattr(status, "status", None) or getattr(status, "state", None)
            client_id = getattr(status, "client_order_id", None) or getattr(status, "client_id", None)

        self.log.info("submit_idempotent: status=%s id=%s", status_desc, client_id)
        return status

    @staticmethod
    def _order_key(order: OrderRequest | OrderStatus | Mapping[str, object]) -> str | None:
        """Return the preferred reconciliation key for an order-like payload."""

        # Support explicit idempotency hints provided either as attributes or via mappings.
        id_fields = ("idempotency_key", "idemp_key", "client_order_id", "client_id")

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
            broker_orders = self._list_orders()
        except NotImplementedError:
            broker_orders = ()

        open_now: dict[str, OrderStatus] = {}
        for broker_order in broker_orders:
            if isinstance(broker_order, Mapping):
                status_value = broker_order.get("status") or broker_order.get("state")
            else:
                status_value = getattr(broker_order, "status", None) or getattr(broker_order, "state", None)

            if isinstance(status_value, str) and status_value.upper() in _TERMINAL_STATUSES:
                continue

            key = self._order_key(broker_order)
            if key is not None:
                open_now[key] = broker_order

        try:
            positions = tuple(self._list_positions())
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

