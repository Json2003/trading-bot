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

    def remaining(self) -> float:
        """Return the amount of quantity that is yet to be filled."""

        return max(self.quantity - self.filled_quantity, 0.0)


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


__all__ = ["Order", "Position", "BrokerBase"]
