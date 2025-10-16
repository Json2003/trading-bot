from __future__ import annotations

import math
from typing import Any, Mapping

import pytest

from brokers.ccxt_broker import CCXTBroker
from tradingbot_core.strategy import OrderIntent


class DummyExchange:
    def __init__(self) -> None:
        self.created: list[Mapping[str, Any]] = []
        self.open_orders_payload: list[Mapping[str, Any]] = []
        self.positions_payload: list[Mapping[str, Any]] | None = []
        self.balance_payload: Mapping[str, Any] = {"total": {}}
        self.cancelled: list[tuple[str, Any, Mapping[str, Any] | None]] = []
        self.raise_on_cancel = False

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        payload = {
            "id": f"order-{len(self.created)}",
            "clientOrderId": params.get("clientOrderId") if params else None,
            "status": "open",
            "symbol": symbol,
            "filled": 0.0,
            "average": None,
        }
        self.created.append(payload)
        return payload

    def fetch_open_orders(self, symbol: str | None = None) -> list[Mapping[str, Any]]:
        return list(self.open_orders_payload)

    def fetch_positions(self) -> list[Mapping[str, Any]] | None:
        if isinstance(self.positions_payload, Exception):
            raise self.positions_payload
        return None if self.positions_payload is None else list(self.positions_payload)

    def fetch_balance(self) -> Mapping[str, Any]:
        return self.balance_payload

    def cancel_order(
        self, order_id: str, symbol: str | None = None, params: Mapping[str, Any] | None = None
    ) -> None:
        self.cancelled.append((order_id, symbol, params))
        if self.raise_on_cancel:
            raise ValueError("cancel failed")


def _make_broker() -> CCXTBroker:
    return CCXTBroker("binance", client=DummyExchange())


def test_intent_to_order_normalises_symbol_and_metadata() -> None:
    broker = _make_broker()
    intent = OrderIntent(idemp_key="k1", symbol="BINANCE:btc-usdt", side="buy", qty=1.5, type="limit", meta=None)

    order = broker.intent_to_order(intent)

    assert order.symbol == "BTC/USDT"
    assert math.isclose(order.quantity, 1.5)
    assert order.idemp_key == "k1"
    assert order.order_type == "limit"
    assert order.metadata == {}


def test_place_creates_order_and_returns_status() -> None:
    broker = _make_broker()
    order = OrderIntent(idemp_key="abc", symbol="btc/usdt", side="buy", qty=0.5, type="limit", meta=None)
    ccxt_order = broker.place(broker.intent_to_order(order))

    assert ccxt_order.status == "open"
    assert ccxt_order.client_id == "abc"
    assert ccxt_order.idemp_key == "abc"
    assert math.isclose(ccxt_order.filled_quantity, 0.0)
    assert ccxt_order.broker_order_id == "order-0"
    assert ccxt_order.raw["symbol"] == "BTC/USDT"


def test_fetch_open_orders_converts_payloads() -> None:
    broker = _make_broker()
    exchange: DummyExchange = broker.client  # type: ignore[assignment]
    exchange.open_orders_payload = [
        {"id": "o-1", "clientOrderId": "k1", "status": "closed", "filled": 1.0, "average": 100.0},
        {"id": "o-2", "clientOrderId": "k2", "status": "open", "filled": 0.0},
    ]

    statuses = broker.fetch_open_orders()

    assert [status.client_id for status in statuses] == ["k1", "k2"]
    assert statuses[0].status == "filled"
    assert math.isclose(statuses[0].filled_quantity, 1.0)
    assert math.isclose(statuses[0].avg_price or 0.0, 100.0)


def test_fetch_positions_prefers_position_endpoint() -> None:
    broker = _make_broker()
    exchange: DummyExchange = broker.client  # type: ignore[assignment]
    exchange.positions_payload = [
        {"symbol": "BTC/USDT", "contracts": 2.0, "entryPrice": 20000.0},
        {"symbol": "ETH/USDT", "positionAmt": -1.0, "avgEntryPrice": 1500.0},
    ]

    positions = broker.fetch_positions()

    assert len(positions) == 2
    assert math.isclose(positions[0]["quantity"], 2.0)
    assert math.isclose(positions[0]["avg_price"] or 0.0, 20000.0)


def test_fetch_positions_falls_back_to_balance() -> None:
    broker = _make_broker()
    exchange: DummyExchange = broker.client  # type: ignore[assignment]
    exchange.positions_payload = None
    exchange.balance_payload = {"total": {"BTC": 1.0, "ETH": 0.0, "USDT": -2.0}}

    positions = broker.fetch_positions()

    assert {pos["symbol"] for pos in positions} == {"BTC", "USDT"}
    btc = next(pos for pos in positions if pos["symbol"] == "BTC")
    assert math.isclose(btc["quantity"], 1.0)


def test_cancel_uses_known_broker_identifier() -> None:
    broker = _make_broker()
    exchange: DummyExchange = broker.client  # type: ignore[assignment]
    order = broker.intent_to_order(OrderIntent(idemp_key="cancel-me", symbol="BTC/USDT", side="buy", qty=1, type="market"))
    status = broker.place(order)

    assert broker.cancel("cancel-me")
    assert exchange.cancelled == [(status.broker_order_id, "BTC/USDT", {})]


def test_cancel_returns_false_on_failure() -> None:
    broker = _make_broker()
    exchange: DummyExchange = broker.client  # type: ignore[assignment]
    exchange.raise_on_cancel = True

    assert not broker.cancel("unknown")
    assert exchange.cancelled == [("unknown", None, {})]
