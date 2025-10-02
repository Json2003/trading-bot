"""Abstract interfaces shared by execution back-ends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol


@dataclass(frozen=True)
class Order:
    """Represents a simplified order used by the reconciler."""

    id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float = 0.0
    status: str = "open"
    price: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def remaining(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_price: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class BrokerBase(Protocol):
    """Protocol describing the methods used by the reconciler."""

    def list_open_orders(self) -> Iterable[Order]:
        ...

    def list_positions(self) -> Iterable[Position]:
        ...


__all__ = ["Order", "Position", "BrokerBase"]
