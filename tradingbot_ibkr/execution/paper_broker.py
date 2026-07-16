"""In-memory broker used for tests and safe paper simulations."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Mapping

from tradingbot_core.strategy import OrderIntent

from .broker_base import BrokerBase, Order, Position


class PaperBroker(BrokerBase):
    """Deterministic paper broker implementing the canonical execution contract."""

    def __init__(self, *, initial_positions: Mapping[str, float] | None = None) -> None:
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {
            symbol: Position(symbol=symbol, quantity=float(qty))
            for symbol, qty in (initial_positions or {}).items()
        }

    def intent_to_order(self, intent: OrderIntent) -> Order:
        """Convert a strategy intent into the shared execution order model."""

        symbol = intent.symbol.split(":", 1)[-1]
        return Order.from_intent(intent, symbol=symbol)

    def submit_order(self, order: Order) -> Order:
        """Submit an order once, keyed by its stable idempotency identifier."""

        key = order.client_order_id or order.idemp_key or order.id
        existing = self._orders.get(str(key))
        if existing is not None:
            return existing

        stored = replace(order, id=str(key))
        self._orders[str(key)] = stored
        return stored

    def fill_order(self, order_id: str, *, filled_quantity: float | None = None) -> Order:
        order = self._orders[order_id]
        qty = order.quantity if filled_quantity is None else min(filled_quantity, order.quantity)
        status = "filled" if qty >= order.quantity else "partially_filled"
        updated = replace(order, filled_quantity=qty, status=status)
        self._orders[order_id] = updated
        self._update_position(updated)
        return updated

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders[order_id]
        cancelled = replace(order, status="cancelled")
        self._orders[order_id] = cancelled
        return cancelled

    def cancel_all_orders(self) -> list[Order]:
        """Cancel every currently open order and return the resulting snapshots."""

        cancelled: list[Order] = []
        for order in list(self.list_open_orders()):
            cancelled.append(self.cancel_order(order.id))
        return cancelled

    def _update_position(self, order: Order) -> None:
        if order.status not in {"filled", "partially_filled"} or order.filled_quantity == 0:
            return
        multiplier = 1 if order.side.lower() == "buy" else -1
        qty_change = multiplier * order.filled_quantity
        position = self._positions.get(order.symbol)
        new_qty = (position.quantity if position else 0.0) + qty_change
        if abs(new_qty) < 1e-9:
            self._positions.pop(order.symbol, None)
        else:
            self._positions[order.symbol] = Position(symbol=order.symbol, quantity=new_qty)

    def list_open_orders(self) -> Iterable[Order]:
        return [
            order
            for order in self._orders.values()
            if order.status in {"open", "partially_filled"}
        ]

    def list_positions(self) -> Iterable[Position]:
        return list(self._positions.values())


__all__ = ["PaperBroker"]
