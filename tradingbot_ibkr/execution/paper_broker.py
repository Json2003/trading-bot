"""In-memory broker used for tests and simple simulations."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Mapping
import uuid

from .broker_base import BrokerBase, Order, Position


class PaperBroker(BrokerBase):
    def __init__(self, *, initial_positions: Mapping[str, float] | None = None) -> None:
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {
            symbol: Position(symbol=symbol, quantity=qty)
            for symbol, qty in (initial_positions or {}).items()
        }

    def submit_order(self, symbol: str, side: str, quantity: float, *, price: float | None = None) -> Order:
        order_id = uuid.uuid4().hex
        order = Order(id=order_id, symbol=symbol, side=side, quantity=quantity, price=price)
        self._orders[order_id] = order
        return order

    def fill_order(self, order_id: str, *, filled_quantity: float | None = None) -> Order:
        order = self._orders[order_id]
        qty = order.quantity if filled_quantity is None else min(filled_quantity, order.quantity)
        updated = replace(order, filled_quantity=qty, status="filled")
        self._orders[order_id] = updated
        self._update_position(updated)
        return updated

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders[order_id]
        cancelled = replace(order, status="cancelled")
        self._orders[order_id] = cancelled
        return cancelled

    def _update_position(self, order: Order) -> None:
        if order.status != "filled" or order.filled_quantity == 0:
            return
        multiplier = 1 if order.side.lower() == "buy" else -1
        qty_change = multiplier * order.filled_quantity
        position = self._positions.get(order.symbol)
        new_qty = (position.quantity if position else 0.0) + qty_change
        if abs(new_qty) < 1e-9:
            self._positions.pop(order.symbol, None)
        else:
            self._positions[order.symbol] = Position(symbol=order.symbol, quantity=new_qty)

    # -- BrokerBase interface -------------------------------------------------
    def list_open_orders(self) -> Iterable[Order]:
        return [order for order in self._orders.values() if order.status == "open"]

    def list_positions(self) -> Iterable[Position]:
        return list(self._positions.values())


__all__ = ["PaperBroker"]
