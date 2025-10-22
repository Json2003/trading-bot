"""Thin adapter over the Interactive Brokers API.

The real project contains a significantly more involved implementation.  For
this kata we only need a minimal class that satisfies the reconciler tests by
conforming to :class:`~tradingbot_ibkr.execution.broker_base.BrokerBase`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .broker_base import BrokerBase, Order, Position


class IBKRBroker(BrokerBase):
    def __init__(
        self, *, orders: Sequence[Order] | None = None, positions: Sequence[Position] | None = None
    ) -> None:
        self._orders = list(orders or [])
        self._positions = list(positions or [])

    def list_open_orders(self) -> Iterable[Order]:
        return [order for order in self._orders if order.status == "open"]

    def list_positions(self) -> Iterable[Position]:
        return list(self._positions)


__all__ = ["IBKRBroker"]
