"""In-memory broker used for tests and safe paper simulations."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Iterable, Mapping

from tradingbot_core.strategy import OrderIntent

from .broker_base import BrokerBase, Order, Position


class PaperBroker(BrokerBase):
    """Deterministic thread-safe broker implementing the execution contract."""

    def __init__(self, *, initial_positions: Mapping[str, float] | None = None) -> None:
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {
            symbol: Position(symbol=symbol, quantity=float(qty))
            for symbol, qty in (initial_positions or {}).items()
        }
        self._lock = RLock()

    def intent_to_order(self, intent: OrderIntent) -> Order:
        """Convert a strategy intent into the shared execution order model."""

        symbol = intent.symbol.split(":", 1)[-1]
        return Order.from_intent(intent, symbol=symbol)

    def submit_order(self, order: Order) -> Order:
        """Submit an order once, keyed by its stable idempotency identifier."""

        key = str(order.client_order_id or order.idemp_key or order.id)
        with self._lock:
            existing = self._orders.get(key)
            if existing is not None:
                return existing

            stored = replace(order, id=key)
            self._orders[key] = stored
            return stored

    def fill_order(self, order_id: str, *, filled_quantity: float | None = None) -> Order:
        """Apply a cumulative fill quantity and update the position by its delta."""

        with self._lock:
            order = self._orders[order_id]
            requested = order.quantity if filled_quantity is None else float(filled_quantity)
            cumulative_qty = max(
                order.filled_quantity,
                min(requested, order.quantity),
            )
            fill_delta = max(cumulative_qty - order.filled_quantity, 0.0)
            status = "filled" if cumulative_qty >= order.quantity else "partially_filled"
            updated = replace(order, filled_quantity=cumulative_qty, status=status)
            self._orders[order_id] = updated
            if fill_delta > 0:
                self._update_position(updated, fill_delta=fill_delta)
            return updated

    def cancel_order(self, order_id: str) -> Order:
        with self._lock:
            order = self._orders[order_id]
            if order.status in {"filled", "cancelled", "rejected"}:
                return order
            cancelled = replace(order, status="cancelled")
            self._orders[order_id] = cancelled
            return cancelled

    def cancel_all_orders(self) -> list[Order]:
        """Cancel every currently open order and return resulting snapshots."""

        with self._lock:
            order_ids = [order.id for order in self._open_orders_unlocked()]
            return [self.cancel_order(order_id) for order_id in order_ids]

    def _update_position(self, order: Order, *, fill_delta: float) -> None:
        multiplier = 1 if order.side.lower() == "buy" else -1
        qty_change = multiplier * fill_delta
        position = self._positions.get(order.symbol)
        new_qty = (position.quantity if position else 0.0) + qty_change
        if abs(new_qty) < 1e-9:
            self._positions.pop(order.symbol, None)
        else:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=new_qty,
                average_price=order.price,
            )

    def _open_orders_unlocked(self) -> list[Order]:
        return [
            order
            for order in self._orders.values()
            if order.status in {"open", "partially_filled"}
        ]

    def list_open_orders(self) -> Iterable[Order]:
        with self._lock:
            return list(self._open_orders_unlocked())

    def list_positions(self) -> Iterable[Position]:
        with self._lock:
            return list(self._positions.values())


__all__ = ["PaperBroker"]
