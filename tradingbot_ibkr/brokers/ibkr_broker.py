"""Production-ready wiring for the Interactive Brokers adapter."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, MutableMapping, Protocol
import logging
import threading
import time
import uuid

from .broker_base import Broker, BrokerEventHandler
from .models import OrderRequest, OrderState, OrderStatus, Position

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from tradingbot_core.monitoring import MonitoringHub

class BrokerClient(Protocol):
    """Protocol describing the client used by :class:`IbkrBroker`."""

    def connect(self) -> None:
        ...

    def submit_order(
        self,
        account_id: str,
        request: OrderRequest,
        *,
        idempotency_key: str,
    ) -> OrderStatus | Mapping[str, object]:
        ...

    def cancel_order(
        self,
        account_id: str,
        broker_order_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> OrderStatus | Mapping[str, object]:
        ...

    def fetch_order(self, account_id: str, broker_order_id: str) -> OrderStatus | Mapping[str, object]:
        ...

    def list_positions(self, account_id: str) -> Iterable[Position | Mapping[str, object]]:
        ...

    def get_cash(self, account_id: str) -> float:
        ...

    def stream_orders(self, account_id: str, handler: Callable[[OrderStatus], None]) -> None:
        ...


def _status_from_payload(payload: OrderStatus | Mapping[str, object]) -> OrderStatus:
    if isinstance(payload, OrderStatus):
        return payload
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        raise TypeError("order status payload must be a mapping or OrderStatus")

    broker_order_id = str(payload.get("broker_order_id") or payload.get("id"))
    raw_state = payload.get("state") or OrderState.NEW
    state = raw_state if isinstance(raw_state, OrderState) else OrderState(str(raw_state))

    return OrderStatus(
        broker_order_id=broker_order_id,
        state=state,
        filled_quantity=float(payload.get("filled_quantity", 0.0)),
        avg_fill_price=payload.get("avg_fill_price"),
        submitted_at=payload.get("submitted_at"),
        updated_at=payload.get("updated_at") or datetime.utcnow(),
        message=payload.get("message"),
        client_order_id=payload.get("client_order_id"),
        symbol=payload.get("symbol"),
    )


def _position_from_payload(payload: Position | Mapping[str, object]) -> Position:
    if isinstance(payload, Position):
        return payload
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        raise TypeError("position payload must be a mapping or Position")

    return Position(
        symbol=str(payload["symbol"]),
        quantity=float(payload.get("quantity", 0.0)),
        avg_price=float(payload.get("avg_price", 0.0)),
        market_price=payload.get("market_price"),
        unrealized_pnl=payload.get("unrealized_pnl"),
        realized_pnl=payload.get("realized_pnl"),
    )


class IbkrBroker(Broker):
    """Interactive Brokers adapter that proxies calls to a user supplied client."""

    name = "ibkr"
    supports_equities = True

    def __init__(
        self,
        base_url: str,
        account_id: str,
        client: BrokerClient,
        *,
        logger: logging.Logger | None = None,
        position_tolerance: float = 1e-6,
        monitor: "MonitoringHub" | None = None,
    ) -> None:
        self.base_url = base_url
        self.account_id = account_id
        self._client = client
        self._logger = logger or logging.getLogger(__name__)
        self._position_tolerance = position_tolerance
        self._expected_positions: MutableMapping[str, float] = {}
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._monitor = monitor

    def connect(self) -> None:
        self._client.connect()

    def _next_idempotency_key(self) -> str:
        return uuid.uuid4().hex

    def _request_payload(self, req: OrderRequest) -> Mapping[str, object]:
        payload = {
            "symbol": req.symbol,
            "quantity": req.quantity,
            "side": req.side.value if hasattr(req.side, "value") else str(req.side),
            "order_type": req.order_type.value if hasattr(req.order_type, "value") else str(req.order_type),
            "time_in_force": req.time_in_force.value if hasattr(req.time_in_force, "value") else str(req.time_in_force),
        }
        if req.limit_price is not None:
            payload["limit_price"] = req.limit_price
        if req.stop_price is not None:
            payload["stop_price"] = req.stop_price
        if req.client_order_id is not None:
            payload["client_order_id"] = req.client_order_id
        return payload

    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        idempotency_key = req.client_order_id or self._next_idempotency_key()
        payload = self._client.submit_order(
            account_id,
            req,
            idempotency_key=idempotency_key,
        )
        status = _status_from_payload(payload)
        if status.client_order_id is None:
            status.client_order_id = idempotency_key
        self._logger.info(
            "Order submitted",
            extra={
                "account_id": account_id,
                "broker_order_id": status.broker_order_id,
                "client_order_id": status.client_order_id,
                "request": self._request_payload(req),
            },
        )
        return status

    def cancel_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        payload = self._client.cancel_order(
            account_id,
            broker_order_id,
            idempotency_key=self._next_idempotency_key(),
        )
        status = _status_from_payload(payload)
        self._logger.info(
            "Order cancellation submitted",
            extra={
                "account_id": account_id,
                "broker_order_id": broker_order_id,
            },
        )
        return status

    def get_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        payload = self._client.fetch_order(account_id, broker_order_id)
        return _status_from_payload(payload)

    def get_positions(self, account_id: str) -> Iterable[Position]:
        for position in self._client.list_positions(account_id):
            yield _position_from_payload(position)

    def get_cash(self, account_id: str) -> float:
        return float(self._client.get_cash(account_id))

    def stream_events(self, on_event: BrokerEventHandler) -> None:
        def handler(status: OrderStatus) -> None:
            resolved = _status_from_payload(status)
            if self._monitor:
                if resolved.state is OrderState.FILLED:
                    if resolved.symbol is not None and resolved.avg_fill_price is not None:
                        self._monitor.record_fill(
                            symbol=resolved.symbol,
                            quantity=resolved.filled_quantity,
                            price=resolved.avg_fill_price,
                        )
                elif resolved.state is OrderState.REJECTED:
                    self._monitor.record_error(
                        error_type="order_rejected",
                        message=resolved.message or "Order rejected",
                    )
            on_event(resolved)

        self._client.stream_orders(self.account_id, handler)

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    def update_expected_positions(self, positions: Mapping[str, float]) -> None:
        self._expected_positions = {self.normalize_symbol(sym): float(qty) for sym, qty in positions.items()}

    def _check_positions_once(self) -> None:
        broker_positions = {pos.symbol: pos.quantity for pos in self.get_positions(self.account_id)}
        discrepancies: dict[str, float] = {}
        for symbol, expected_qty in self._expected_positions.items():
            broker_qty = broker_positions.get(symbol, 0.0)
            delta = broker_qty - expected_qty
            if abs(delta) > self._position_tolerance:
                discrepancies[symbol] = delta
        extra_unknown = {
            symbol: qty
            for symbol, qty in broker_positions.items()
            if symbol not in self._expected_positions and abs(qty) > self._position_tolerance
        }
        if discrepancies or extra_unknown:
            self._logger.warning(
                "Position sanity check detected mismatches",
                extra={
                    "expected": dict(self._expected_positions),
                    "broker": broker_positions,
                    "discrepancies": discrepancies,
                    "unexpected": extra_unknown,
                },
            )

    def start_position_monitor(
        self,
        *,
        interval: float = 60.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():  # pragma: no cover - idempotent guard
            return

        self._monitor_stop.clear()

        def _loop() -> None:
            while not self._monitor_stop.is_set():
                try:
                    self._check_positions_once()
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._logger.exception("Position sanity check failed", exc_info=exc)
                    if self._monitor:
                        self._monitor.record_error(
                            error_type="position_monitor",
                            message=str(exc),
                        )
                if self._monitor_stop.is_set():
                    break
                sleeper(interval)

        self._monitor_thread = threading.Thread(target=_loop, name="ibkr-position-monitor", daemon=True)
        self._monitor_thread.start()

    def stop_position_monitor(self) -> None:
        if not self._monitor_thread:
            return
        self._monitor_stop.set()
        self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None


__all__ = ["IbkrBroker", "BrokerClient"]
