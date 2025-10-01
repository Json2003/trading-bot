"""Utilities for routing order requests to the correct broker implementation."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Protocol, Sequence, Tuple, runtime_checkable

from .models import OrderRequest


class RouterError(RuntimeError):
    """Base exception for router issues."""


class RouterNotConfiguredError(RouterError):
    """Raised when an operation is attempted without any configured broker."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "No brokers have been configured on the router.")


class UnknownBrokerError(RouterError):
    """Raised when a requested broker alias does not exist."""

    def __init__(self, broker: str, *, available: Sequence[str] | None = None) -> None:
        if available:
            available_brokers = ", ".join(sorted(available)) or "<none>"
            message = f"Unknown broker '{broker}'. Available brokers: {available_brokers}."
        else:
            message = f"Unknown broker '{broker}'."
        super().__init__(message)
        self.broker = broker
        self.available = tuple(available) if available is not None else None


@runtime_checkable
class SupportsToDict(Protocol):
    """Protocol describing objects that can be serialized via ``to_dict``."""

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - protocol definition
        """Return a dictionary representation of the object."""


class BrokerLike(Protocol):
    """Structural protocol describing the broker operations the router requires."""

    def place_order(self, account_id: str, req: OrderRequest) -> Any:  # pragma: no cover - protocol definition
        """Submit an order to the broker."""

    def get_positions(self, account_id: str) -> Iterable[Any]:  # pragma: no cover - protocol definition
        """Return the open positions for ``account_id``."""

    def list_accounts(self) -> Iterable[str]:  # pragma: no cover - protocol definition
        """Return the known account identifiers."""


class OrderRouter:
    """Route order actions to concrete broker implementations."""

    brokers: MutableMapping[str, BrokerLike]
    default: str | None

    def __init__(
        self,
        brokers: Mapping[str, BrokerLike] | None = None,
        *,
        default: str | None = None,
    ) -> None:
        self.brokers = {}
        self.default = None
        if brokers:
            self.configure(brokers, default=default)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def configure(
        self,
        brokers: Mapping[str, BrokerLike],
        *,
        default: str | None = None,
    ) -> None:
        """Replace the currently configured brokers.

        Args:
            brokers: Mapping of broker alias to broker implementation.
            default: Optional alias to use as the default broker. If omitted the
                first item from ``brokers`` is used.
        """

        if not brokers:
            raise RouterNotConfiguredError("Cannot configure router without brokers.")

        self.brokers = dict(brokers)
        if default is None:
            default = next(iter(self.brokers))
        if default not in self.brokers:
            raise UnknownBrokerError(default, available=tuple(self.brokers))
        self.default = default

    def register(self, name: str, broker: BrokerLike, *, default: bool = False) -> None:
        """Register ``broker`` under ``name``.

        If ``default`` is true the broker becomes the default target.
        """

        self.brokers[name] = broker
        if self.default is None or default:
            self.default = name

    # ------------------------------------------------------------------
    # Operational helpers
    # ------------------------------------------------------------------
    def _resolve_broker(self, broker_hint: str | None) -> Tuple[str, BrokerLike]:
        """Return the broker name and instance for ``broker_hint``.

        Falls back to the default broker when ``broker_hint`` is ``None``.
        """

        name = broker_hint or self.default
        if name is None:
            raise RouterNotConfiguredError("No default broker configured.")
        try:
            return name, self.brokers[name]
        except KeyError as exc:
            raise UnknownBrokerError(name, available=tuple(self.brokers)) from exc

    def place(
        self,
        account_id: str,
        req: OrderRequest,
        *,
        broker_hint: str | None = None,
    ) -> Any:
        """Submit an order request to the resolved broker."""

        _, broker = self._resolve_broker(broker_hint)
        result = broker.place_order(account_id, req)
        return self._serialize(result)

    def positions(
        self,
        account_id: str,
        *,
        broker_hint: str | None = None,
    ) -> List[Any]:
        """Return serialized positions for ``account_id`` from the resolved broker."""

        _, broker = self._resolve_broker(broker_hint)
        positions = broker.get_positions(account_id)
        return [self._serialize(position) for position in positions]

    def accounts(self) -> List[Dict[str, Any]]:
        """Return metadata about the configured brokers and their accounts."""

        payload: List[Dict[str, Any]] = []
        for name, broker in self.brokers.items():
            accounts = self._collect_accounts(broker)
            payload.append(
                {
                    "broker": name,
                    "accounts": accounts,
                    "is_default": name == self.default,
                }
            )
        payload.sort(key=lambda item: item["broker"])
        return payload

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _collect_accounts(self, broker: BrokerLike) -> List[str]:
        raw_accounts: Iterable[str]
        try:
            raw_accounts = broker.list_accounts()
        except AttributeError:
            return []
        normalized = {str(account) for account in raw_accounts}
        return sorted(normalized)

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, SupportsToDict):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [self._serialize(item) for item in value]
        return value


__all__ = [
    "OrderRouter",
    "RouterError",
    "RouterNotConfiguredError",
    "UnknownBrokerError",
]
