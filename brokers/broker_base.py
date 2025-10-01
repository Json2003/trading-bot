"""Abstract base class describing a trading broker interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from models import OrderRequest, OrderStatus, Position


class Broker(ABC):
    """Base contract for broker integrations.

    Concrete implementations should override the abstract methods and provide
    useful defaults for the optional ones.  The base class intentionally keeps
    the surface area small so that new brokers can be implemented without
    having to support every feature up front.
    """

    name: str = "broker"
    supports_crypto: bool = False
    paper_trading: bool = False

    def connect(self) -> None:  # pragma: no cover - optional hook
        """Establish a connection with the underlying broker service."""
        # Implementations may override if they need a pre-flight check.
        return None

    def normalize_symbol(self, symbol: str) -> str:  # pragma: no cover - passthrough
        """Return a broker-compatible representation of ``symbol``."""
        return symbol

    @abstractmethod
    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        """Submit an order request to the broker and return its status."""

    def cancel_order(self, account_id: str, broker_order_id: str) -> bool:
        """Cancel an order if the broker supports it.

        Implementations should return ``True`` if the cancellation request was
        acknowledged.  The default implementation raises ``NotImplementedError``
        so that integrators are aware of the missing capability.
        """

        raise NotImplementedError

    def get_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        """Fetch the latest status for an order."""

        raise NotImplementedError

    def list_orders(self, account_id: str) -> Sequence[OrderStatus]:  # pragma: no cover
        """Return known orders for the account.

        Implementations may provide this convenience method to support batch
        synchronisation.  The default implementation raises
        ``NotImplementedError`` to avoid silently returning stale data.
        """

        raise NotImplementedError

    def get_positions(self, account_id: str) -> Sequence[Position]:
        """Return open positions for the account."""

        raise NotImplementedError

    def get_cash(self, account_id: str) -> float:
        """Return the amount of available cash on the account."""

        raise NotImplementedError

    def stream_events(
        self,
        handler: Callable[[Dict[str, Any]], None],
        *,
        channels: Optional[Iterable[str]] = None,
    ) -> None:  # pragma: no cover - streaming is environment specific
        """Start streaming events from the broker.

        ``handler`` receives a dictionary with the event payload.  Streaming is
        highly broker-specific and therefore optional; implementers may raise
        ``NotImplementedError`` if their broker does not support it.
        """

        raise NotImplementedError
