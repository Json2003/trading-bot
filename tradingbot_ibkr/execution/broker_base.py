"""Shared execution data structures and the canonical broker protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Protocol

if TYPE_CHECKING:
    from tradingbot_core.strategy import OrderIntent


@dataclass(frozen=True, slots=True)
class Order:
    """Broker-neutral order tracked by execution, reconciliation and risk layers."""

    id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float = 0.0
    status: str = "open"
    price: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    client_order_id: str | None = None
    idemp_key: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None

    def remaining(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def qty(self) -> float:
        return self.quantity

    @classmethod
    def from_intent(
        cls,
        intent: OrderIntent,
        *,
        order_id: str | None = None,
        symbol: str | None = None,
    ) -> Order:
        """Create an order from a strategy-layer ``OrderIntent``."""

        try:
            meta: Mapping[str, object] = intent.meta
        except AttributeError:  # pragma: no cover - defensive programming
            meta = {}

        order_meta: dict[str, object] = {"intent": intent}
        if isinstance(meta, Mapping):
            order_meta.update(meta)

        resolved_id = order_id or getattr(intent, "idemp_key", None)
        if resolved_id is None:
            import uuid

            resolved_id = uuid.uuid4().hex

        if isinstance(meta, Mapping) and meta.get("client_order_id"):
            client_order_id = str(meta["client_order_id"])
        else:
            idemp_key = getattr(intent, "idemp_key", None)
            client_order_id = str(idemp_key) if idemp_key else None

        return cls(
            id=str(resolved_id),
            symbol=symbol or getattr(intent, "symbol", ""),
            side=str(getattr(intent, "side", "")),
            quantity=float(getattr(intent, "qty", 0.0)),
            price=getattr(intent, "limit_price", None),
            metadata=order_meta,
            client_order_id=client_order_id,
            idemp_key=getattr(intent, "idemp_key", None),
            order_type=(
                str(getattr(intent, "type", ""))
                if getattr(intent, "type", None)
                else None
            ),
            time_in_force=(
                str(meta.get("time_in_force"))
                if isinstance(meta, Mapping) and meta.get("time_in_force")
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class OrderStatus:
    """Broker acknowledgement or fill snapshot."""

    client_id: str | None
    exchange_id: str | None
    status: str
    filled_qty: float = 0.0
    avg_price: float | None = None
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Position:
    """Broker-neutral position snapshot."""

    symbol: str
    quantity: float
    average_price: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class BrokerBase(Protocol):
    """Canonical execution contract implemented by paper and live adapters."""

    def submit_order(self, order: Order) -> Order | OrderStatus:
        """Submit one canonical order and return its broker snapshot."""

    def cancel_order(self, order_id: str) -> Order | OrderStatus | bool:
        """Cancel one order by its stable local or broker identifier."""

    def cancel_all_orders(self) -> Iterable[Order | OrderStatus | str]:
        """Cancel all currently open orders."""

    def list_open_orders(self) -> Iterable[Order]:
        """Return all currently open or partially filled orders."""

    def list_positions(self) -> Iterable[Position]:
        """Return all non-flat positions."""


__all__ = ["Order", "OrderStatus", "Position", "BrokerBase"]
