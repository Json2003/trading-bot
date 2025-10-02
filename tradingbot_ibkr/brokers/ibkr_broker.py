"""Thin Interactive Brokers (IBKR) adapter used in tests.

The production codebase contains a much richer implementation that talks to
either the Trader Workstation API or IBKR's REST endpoints.  The unit tests in
this kata only need a lightweight stand-in so that other components (such as
the order router) can be exercised without pulling in the heavy ib_insync
dependency.

The class defined here purposefully keeps the surface area minimal while still
mirroring the public contract of the real integration.  The operational
methods raise :class:`NotImplementedError` to make it explicit that the stub is
not wired to a live broker; the reconciler tests monkeypatch these methods with
in-memory fakes when required.
"""
from __future__ import annotations

from typing import Iterable

from .broker_base import Broker, BrokerEventHandler
from .models import OrderRequest, OrderStatus, Position


class IbkrBroker(Broker):
    """Skeleton IBKR broker that records connection parameters.

    Parameters
    ----------
    base_url:
        Base URL of the IBKR REST gateway.  The real adapter uses this to
        dispatch HTTP requests.
    account_id:
        Account identifier that orders should be associated with.
    """

    name = "ibkr"
    supports_equities = True

    def __init__(self, base_url: str, account_id: str) -> None:
        self.base_url = base_url
        self.account_id = account_id

    # The abstract base class defines a fairly rich interface.  We keep the
    # implementations intentionally unimplemented so tests can provide their
    # own behaviour through monkeypatching without pulling real network calls
    # into the test suite.
    def connect(self) -> None:  # pragma: no cover - network side effects
        raise NotImplementedError("IBKR connect routine is not implemented in tests")

    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        raise NotImplementedError("Order placement is not implemented for the IBKR stub")

    def cancel_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        raise NotImplementedError("Order cancellation is not implemented for the IBKR stub")

    def get_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        raise NotImplementedError("Order retrieval is not implemented for the IBKR stub")

    def get_positions(self, account_id: str) -> Iterable[Position]:
        raise NotImplementedError("Position fetching is not implemented for the IBKR stub")

    def get_cash(self, account_id: str) -> float:
        raise NotImplementedError("Cash retrieval is not implemented for the IBKR stub")

    def stream_events(self, on_event: BrokerEventHandler) -> None:  # pragma: no cover - async IO
        raise NotImplementedError("Event streaming is not implemented for the IBKR stub")

    def normalize_symbol(self, symbol: str) -> str:
        # Mirror the production behaviour of upper-casing symbols while
        # preserving separators.
        return symbol.strip().upper()


__all__ = ["IbkrBroker"]
