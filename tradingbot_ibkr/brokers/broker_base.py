"""Abstract base class defining the broker integration contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable

from .models import OrderRequest, OrderStatus, Position

BrokerEventHandler = Callable[[OrderStatus], None]


class Broker(ABC):
    """Interface that concrete broker adapters must implement."""

    name: str
    supports_crypto: bool = False
    supports_equities: bool = True
    supports_options: bool = False
    supports_futures: bool = False
    paper_trading: bool = False

    @abstractmethod
    def connect(self) -> None:
        """Establish any network connections or authentication sessions."""

    @abstractmethod
    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        """Submit an order request and return its initial status."""

    @abstractmethod
    def cancel_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        """Attempt to cancel an outstanding order."""

    @abstractmethod
    def get_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        """Fetch the latest status for a given order."""

    @abstractmethod
    def get_positions(self, account_id: str) -> Iterable[Position]:
        """Return all open positions for the provided account."""

    @abstractmethod
    def get_cash(self, account_id: str) -> float:
        """Return the available cash balance in the account's base currency."""

    @abstractmethod
    def stream_events(self, on_event: BrokerEventHandler) -> None:
        """Stream order updates to the provided callback."""

    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Convert user provided symbols to the broker specific format."""
