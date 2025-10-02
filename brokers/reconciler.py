"""Utilities for reconciling intended orders with broker state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .broker_base import Broker, Order, OrderStatus


class LoggerLike(Protocol):
    """Protocol describing the subset of :mod:`logging` used by the reconciler."""

    def info(self, msg: str, *args, **kwargs) -> None:  # pragma: no cover - protocol stub
        ...

    def error(self, msg: str, *args, **kwargs) -> None:  # pragma: no cover - protocol stub
        ...


@dataclass(slots=True)
class RiskLimits:
    """Simple container describing risk parameters for the reconciler."""

    max_daily_loss_pct: float
    kill_switch_drawdown_pct: float
    max_position_risk_pct: float


class Reconciler:
    """Compare the strategy's intent with the broker state and react accordingly."""

    def __init__(self, broker: Broker, limits: RiskLimits, logger: LoggerLike):
        self.broker = broker
        self.limits = limits
        self.log = logger

    # ------------------------------------------------------------------
    # Helpers
    def _order_key(self, order: object) -> str | None:
        for attr in ("idemp_key", "client_id", "id", "broker_order_id"):
            value = getattr(order, attr, None)
            if value:
                return str(value)
        return None

    def _open_orders(self) -> dict[str, object]:
        open_now: dict[str, object] = {}
        for existing in self.broker.fetch_open_orders():
            key = self._order_key(existing)
            if key is None:
                continue
            # Always keep the freshest snapshot for the key.  Brokers occasionally
            # return duplicates and we want later entries to win as they tend to be
            # newer.
            open_now[key] = existing
        return open_now

    def _coerce_status(self, payload: object, fallback_key: str | None) -> OrderStatus:
        if isinstance(payload, OrderStatus):
            return payload
        status = getattr(payload, "status", "open")
        client_id = getattr(payload, "client_id", None)
        idemp_key = getattr(payload, "idemp_key", fallback_key)
        filled = getattr(payload, "filled_quantity", 0.0)
        avg_price = getattr(payload, "avg_price", None)
        message = getattr(payload, "message", None)
        return OrderStatus(
            status=status,
            client_id=client_id,
            idemp_key=idemp_key,
            filled_quantity=filled,
            avg_price=avg_price,
            message=message,
        )

    # ------------------------------------------------------------------
    # Public API
    def submit_idempotent(
        self,
        order: Order,
        *,
        open_orders: dict[str, object] | None = None,
    ) -> OrderStatus:
        """Submit ``order`` while avoiding accidental duplicates."""

        key = self._order_key(order)
        current_open = open_orders if open_orders is not None else self._open_orders()
        if key and key in current_open:
            status = self._coerce_status(current_open[key], key)
            self.log.info(
                "submit_idempotent: reusing existing order status=%s id=%s",
                status.status,
                status.client_id,
            )
            return status

        placed = self.broker.place(order)
        status = self._coerce_status(placed, key)
        if key and open_orders is not None:
            open_orders[key] = status
        self.log.info(
            "submit_idempotent: submitted order status=%s id=%s",
            status.status,
            status.client_id,
        )
        return status

    def reconcile(self, intended_orders: Iterable[Order]) -> None:
        """Ensure every intended order exists at the broker exactly once."""

        open_now = self._open_orders()
        positions: Sequence[object] = tuple(self.broker.fetch_positions())
        self.log.info(
            "reconcile: open=%d positions=%s",
            len(open_now),
            positions,
        )

        for order in intended_orders:
            key = self._order_key(order)
            if key is None or key not in open_now:
                self.submit_idempotent(order, open_orders=open_now)

    def check_kill_switch(self, equity_curve: Sequence[float]) -> bool:
        """Return ``True`` if the kill switch should halt trading."""

        if not equity_curve:
            return False

        latest = float(equity_curve[-1])
        peak = max(float(value) for value in equity_curve)
        if peak > 0:
            drawdown_pct = 100.0 * (1 - latest / peak)
        else:
            drawdown_pct = 0.0

        if drawdown_pct >= self.limits.kill_switch_drawdown_pct:
            self.log.error(
                "KILL SWITCH TRIGGERED: drawdown %.2f%% (limit %.2f%%)",
                drawdown_pct,
                self.limits.kill_switch_drawdown_pct,
            )
            return True

        start = float(equity_curve[0])
        if start > 0:
            loss_pct = 100.0 * (start - latest) / start
            if loss_pct >= self.limits.max_daily_loss_pct:
                self.log.error(
                    "KILL SWITCH TRIGGERED: daily loss %.2f%% (limit %.2f%%)",
                    loss_pct,
                    self.limits.max_daily_loss_pct,
                )
                return True

        return False


__all__ = ["RiskLimits", "Reconciler"]
