"""Tests for the lightweight ccxt execution broker wrapper."""

from __future__ import annotations

from tradingbot_core.strategy import OrderIntent

from tradingbot_ibkr.execution.ccxt_broker import CCXTBroker


class DummyExchange:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, str, float, float | None, dict[str, object]]] = []
        self.cancelled: list[tuple[str, str | None]] = []
        self.orders: dict[str, dict[str, object]] = {}

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        qty: float,
        price: float | None,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        params = params or {}
        client_order_id = params.get("clientOrderId")
        order_id = client_order_id or f"ex-{len(self.orders) + 1}"
        payload = {
            "id": order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side,
            "amount": qty,
            "status": "open",
            "filled": 0.0,
            "average": price,
        }
        self.orders[str(order_id)] = payload
        self.created.append((symbol, order_type, side, qty, price, dict(params)))
        return payload

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, object]]:
        if symbol is None:
            return list(self.orders.values())
        return [order for order in self.orders.values() if order["symbol"] == symbol]

    def fetch_positions(self) -> list[dict[str, object]]:
        return [
            {"symbol": "BTC/USDT", "positionAmt": 0.1, "entryPrice": 101.5},
        ]

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, object]:
        if order_id not in self.orders:
            raise ValueError("unknown order")
        self.cancelled.append((order_id, symbol))
        self.orders.pop(order_id, None)
        return {"id": order_id}


def build_intent(limit_price: float | None = 10.5) -> OrderIntent:
    return OrderIntent(
        idemp_key="intent-1",
        symbol="BINANCE:BTC/USDT",
        side="buy",
        qty=0.25,
        type="limit" if limit_price is not None else "market",
        limit_price=limit_price,
    )


def test_intent_to_order_maps_symbol_and_metadata() -> None:
    broker = CCXTBroker(client=DummyExchange())
    intent = build_intent()

    order = broker.intent_to_order(intent)

    assert order.id == "intent-1"
    assert order.symbol == "BTC/USDT"
    assert order.quantity == 0.25
    assert order.order_type == "limit"
    assert order.price == 10.5
    assert order.client_order_id == "intent-1"
    assert order.metadata["intent"] is intent


def test_place_returns_status_from_exchange_payload() -> None:
    exchange = DummyExchange()
    broker = CCXTBroker(client=exchange)
    order = broker.intent_to_order(build_intent())

    status = broker.place(order)

    assert exchange.created  # ensure we hit the exchange client
    assert status.client_id == "intent-1"
    assert status.exchange_id == "intent-1"
    assert status.status == "open"
    assert status.filled_qty == 0.0
    assert status.avg_price == 10.5


def test_fetch_open_orders_and_cancel_order() -> None:
    exchange = DummyExchange()
    broker = CCXTBroker(client=exchange)
    order = broker.intent_to_order(build_intent())
    broker.place(order)

    open_orders = broker.fetch_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "BTC/USDT"

    positions = broker.fetch_positions()
    assert positions == {"BTC/USDT": 0.1}

    assert broker.cancel("intent-1") is True
    assert exchange.cancelled == [("intent-1", "BTC/USDT")]
