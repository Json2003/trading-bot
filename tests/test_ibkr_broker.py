from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable

import pytest

from tradingbot_ibkr.brokers.ibkr_broker import IbkrBroker, BrokerClient
from tradingbot_ibkr.brokers.models import OrderRequest, OrderSide, OrderState, OrderStatus, Position


class DummyClient(BrokerClient):
    def __init__(self, iteration_event: threading.Event | None = None) -> None:
        self.submit_keys: list[str] = []
        self.position_calls = 0
        self.positions: list[Position] = [Position(symbol="BTC", quantity=1.5, avg_price=25000)]
        self._iteration_event = iteration_event

    def connect(self) -> None:  # pragma: no cover - simple stub
        self.connected = True

    def submit_order(self, account_id: str, request: OrderRequest, *, idempotency_key: str) -> OrderStatus:
        self.submit_keys.append(idempotency_key)
        return OrderStatus(
            broker_order_id="brk-1",
            state=OrderState.NEW,
            client_order_id=idempotency_key,
            symbol=request.symbol,
        )

    def cancel_order(
        self,
        account_id: str,
        broker_order_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> OrderStatus:
        return OrderStatus(
            broker_order_id=broker_order_id,
            state=OrderState.CANCELED,
            client_order_id=idempotency_key,
            symbol="ETH",
        )

    def fetch_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        return OrderStatus(
            broker_order_id=broker_order_id,
            state=OrderState.FILLED,
            filled_quantity=1.0,
            symbol="ETH",
        )

    def list_positions(self, account_id: str) -> Iterable[Position]:
        self.position_calls += 1
        if self._iteration_event is not None:
            self._iteration_event.set()
        return list(self.positions)

    def get_cash(self, account_id: str) -> float:
        return 1000.0

    def stream_orders(self, account_id: str, handler: Callable[[OrderStatus], None]) -> None:
        handler(
            OrderStatus(
                broker_order_id="brk-stream",
                state=OrderState.FILLED,
                filled_quantity=1.0,
                avg_fill_price=25000.0,
                symbol="ETH",
            )
        )


class DummyMonitor:
    def __init__(self) -> None:
        self.fills: list[tuple[str, float, float]] = []
        self.errors: list[tuple[str, str]] = []

    def record_fill(self, *, symbol: str, quantity: float, price: float) -> None:
        self.fills.append((symbol, quantity, price))

    def record_error(self, *, error_type: str, message: str) -> None:
        self.errors.append((error_type, message))

    def record_kill_switch(self, **_: object) -> None:  # pragma: no cover - not used here
        pass


def test_ibkr_broker_uses_client_order_id_for_idempotency() -> None:
    client = DummyClient()
    broker = IbkrBroker("https://example", "acct", client)
    request = OrderRequest(symbol="ETH", quantity=2.0, side=OrderSide.BUY, client_order_id="custom-123")

    status = broker.place_order("acct", request)

    assert client.submit_keys == ["custom-123"]
    assert status.client_order_id == "custom-123"


def test_ibkr_broker_generates_idempotency_key() -> None:
    client = DummyClient()
    broker = IbkrBroker("https://example", "acct", client)
    request = OrderRequest(symbol="ETH", quantity=2.0, side=OrderSide.SELL)

    status = broker.place_order("acct", request)

    assert len(client.submit_keys) == 1
    assert status.client_order_id == client.submit_keys[0]
    assert status.client_order_id is not None


def test_stream_events_feed_monitor() -> None:
    client = DummyClient()
    monitor = DummyMonitor()
    broker = IbkrBroker("https://example", "acct", client, monitor=monitor)
    events: list[OrderStatus] = []

    broker.stream_events(events.append)

    assert monitor.fills == [("ETH", 1.0, 25000.0)]
    assert events and events[0].broker_order_id == "brk-stream"


def test_position_monitor_runs_sanity_checks(caplog: pytest.LogCaptureFixture) -> None:
    iteration = threading.Event()
    client = DummyClient(iteration)
    broker = IbkrBroker("https://example", "acct", client, logger=logging.getLogger("test.ibkr"))
    broker.update_expected_positions({"BTC": 1.0})

    with caplog.at_level(logging.WARNING, logger="test.ibkr"):
        broker.start_position_monitor(interval=0.0, sleeper=lambda _: None)
        assert iteration.wait(timeout=1.0)
    broker.stop_position_monitor()

    assert client.position_calls >= 1
    assert any("Position sanity check" in record.message for record in caplog.records)
