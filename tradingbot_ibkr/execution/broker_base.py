"""Foundational data structures and protocols for the execution layer.

The production code base contains substantially richer models, however the
exercises in this kata only rely on a narrow subset of that functionality.  The
lightweight implementations below focus on capturing the behaviour required by
the reconciler tests while remaining convenient to use in the in-memory paper
broker used throughout the suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class Order:
    """Minimal representation of an order tracked by the reconciler.

    The object purposely omits broker specific attributes and only models the
    small set of fields that the tests exercise.  ``metadata`` acts as an escape
    hatch so callers can attach additional context without having to extend the
    dataclass.
    """

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
        """Return the amount of quantity that is yet to be filled."""

        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def qty(self) -> float:
        """Alias for :attr:`quantity` mirroring ccxt nomenclature."""

        return self.quantity

    @classmethod
    def from_intent(
        cls,
        intent: "OrderIntent",
        *,
        order_id: str | None = None,
        symbol: str | None = None,
    ) -> "Order":
        """Create an :class:`Order` from an :class:`~tradingbot_core.strategy.OrderIntent`.

        The reconciler primarily operates on :class:`Order` objects but the
        strategy layer emits :class:`OrderIntent` instances.  This helper bridges
        the two representations while preserving useful metadata such as the
        idempotency key and any custom parameters included in ``intent.meta``.
        """

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
            order_type=str(getattr(intent, "type", "")) if getattr(intent, "type", None) else None,
            time_in_force=str(meta.get("time_in_force")) if isinstance(meta, Mapping) and meta.get("time_in_force") else None,
        )


@dataclass(frozen=True, slots=True)
class OrderStatus:
    """Lightweight snapshot of a broker order."""

    client_id: str | None
    exchange_id: str | None
    status: str
    filled_qty: float = 0.0
    avg_price: float | None = None
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Position:
    """Simplified view of a broker position."""

    symbol: str
    quantity: float
    average_price: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class BrokerBase(Protocol):
    """Protocol describing the minimal broker surface consumed by the reconciler."""

    def list_open_orders(self) -> Iterable[Order]:
        """Return the currently open orders known to the broker."""

    def list_positions(self) -> Iterable[Position]:
        """Return the positions tracked by the broker."""


__all__ = ["Order", "OrderStatus", "Position", "BrokerBase"]
