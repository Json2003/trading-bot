"""Utilities to compare local state with broker state."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping

from .broker_base import BrokerBase, Order, Position


@dataclass(frozen=True)
class ReconciliationReport:
    missing_orders: tuple[str, ...] = ()
    unexpected_orders: tuple[Order, ...] = ()
    quantity_mismatches: Mapping[str, float] = field(default_factory=dict)
    position_deltas: Mapping[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_clean(self) -> bool:
        return not (self.missing_orders or self.unexpected_orders or self.quantity_mismatches or self.position_deltas)


class Reconciler:
    def __init__(self, broker: BrokerBase, *, quantity_tolerance: float = 1e-6) -> None:
        self._broker = broker
        self._quantity_tolerance = quantity_tolerance

    def _coerce_orders(self, orders: Iterable[Order] | Mapping[str, Order]) -> MutableMapping[str, Order]:
        if isinstance(orders, Mapping):
            return dict(orders)
        return {order.id: order for order in orders}

    def _coerce_positions(self, positions: Mapping[str, float] | Iterable[Position]) -> MutableMapping[str, float]:
        if isinstance(positions, Mapping):
            return dict(positions)
        return {pos.symbol: pos.quantity for pos in positions}

    def reconcile(
        self,
        *,
        local_orders: Iterable[Order] | Mapping[str, Order],
        local_positions: Mapping[str, float] | Iterable[Position],
    ) -> ReconciliationReport:
        broker_orders = {order.id: order for order in self._broker.list_open_orders()}
        broker_positions = {position.symbol: position.quantity for position in self._broker.list_positions()}

        local_orders_map = self._coerce_orders(local_orders)
        local_positions_map = self._coerce_positions(local_positions)

        missing_orders = tuple(sorted(order_id for order_id in local_orders_map.keys() - broker_orders.keys()))
        unexpected_orders = tuple(broker_orders[oid] for oid in broker_orders.keys() - local_orders_map.keys())

        quantity_mismatches: dict[str, float] = {}
        for order_id in broker_orders.keys() & local_orders_map.keys():
            broker_order = broker_orders[order_id]
            local_order = local_orders_map[order_id]
            delta = broker_order.remaining() - local_order.remaining()
            if abs(delta) > self._quantity_tolerance:
                quantity_mismatches[order_id] = delta

        position_deltas: dict[str, float] = {}
        symbols = broker_positions.keys() | local_positions_map.keys()
        for symbol in symbols:
            broker_qty = broker_positions.get(symbol, 0.0)
            local_qty = local_positions_map.get(symbol, 0.0)
            delta = broker_qty - local_qty
            if abs(delta) > self._quantity_tolerance:
                position_deltas[symbol] = delta

        return ReconciliationReport(
            missing_orders=missing_orders,
            unexpected_orders=unexpected_orders,
            quantity_mismatches=quantity_mismatches,
            position_deltas=position_deltas,
        )


__all__ = ["ReconciliationReport", "Reconciler"]
