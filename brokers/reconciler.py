"""Utilities for reconciling intended orders with broker state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Order(Protocol):
    """Protocol describing the subset of order attributes we rely on."""

    client_id: str | None
    idemp_key: str | None


@runtime_checkable
class OrderStatus(Protocol):
    """Protocol describing the status returned by the broker."""

    status: str
    client_id: str | None


@runtime_checkable
class Broker(Protocol):
    """Protocol for the broker primitives used during reconciliation."""

    def place(self, order: Order) -> OrderStatus: ...

    def fetch_open_orders(self) -> Iterable[Order]: ...

    def fetch_positions(self) -> Sequence[object]: ...


@dataclass(slots=True)
class RiskLimits:
    """Configuration for broker level risk controls."""

    max_daily_loss_pct: float
    kill_switch_drawdown_pct: float
    max_position_risk_pct: float


class Reconciler:
    """Synchronise the intended orders with the broker state."""

    def __init__(self, broker: Broker, limits: RiskLimits, logger) -> None:
        self.broker = broker
        self.limits = limits
        self.log = logger

    def submit_idempotent(self, order: Order) -> OrderStatus:
        """Submit ``order`` while avoiding duplicate broker entries."""

        status = self.broker.place(order)
        self.log.info("submit_idempotent: status=%s id=%s", status.status, status.client_id)
        return status

    @staticmethod
    def _order_key(order: Order) -> str | None:
        """Return the preferred reconciliation key for an order."""

        key = getattr(order, "idemp_key", None)
        if key:
            return key
        return getattr(order, "client_id", None)

    def reconcile(self, intended_orders: Iterable[Order]) -> None:
        """Compare intended state vs broker state and heal any drift."""

        open_now = {}
        for broker_order in self.broker.fetch_open_orders():
            key = self._order_key(broker_order)
            if key is not None:
                open_now[key] = broker_order

        positions = self.broker.fetch_positions()
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


__all__ = ["RiskLimits", "Reconciler", "Broker", "Order", "OrderStatus"]

