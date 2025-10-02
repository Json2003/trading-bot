from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from brokers.broker_base import Broker
from brokers.reconciler import Reconciler, RiskLimits
from models import OrderRequest, OrderStatus, OrderType, Side, TimeInForce


def _order_request(
    *,
    client_order_id: str | None,
    symbol: str = "AAPL",
    side: Side = Side.BUY,
    qty: float = 1.0,
    order_type: OrderType = OrderType.MARKET,
    tif: TimeInForce = TimeInForce.DAY,
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        tif=tif,
        client_order_id=client_order_id,
    )


@dataclass
class FakeBroker(Broker):
    open_orders: List[OrderStatus] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    submitted: List[OrderRequest] = field(default_factory=list)

    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        self.submitted.append(req)
        return OrderStatus(
            broker="fake",
            broker_order_id=f"order-{len(self.submitted)}",
            client_order_id=req.client_order_id,
            status="NEW",
            filled_qty=0.0,
            avg_price=None,
            ts=0.0,
            raw={},
        )

    def list_orders(self, account_id: str) -> Iterable[OrderStatus]:
        return list(self.open_orders)

    def get_positions(self, account_id: str) -> Iterable[str]:
        return list(self.positions)


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[object, ...]]] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append((message, args))

    def error(self, message: str, *args: object) -> None:
        self.messages.append((message, args))


def test_reconcile_ignores_orders_with_same_idempotency_key() -> None:
    broker = FakeBroker(
        open_orders=[
            OrderStatus(
                broker="fake",
                broker_order_id="open-1",
                client_order_id="abc",
                status="NEW",
                filled_qty=0.0,
                avg_price=None,
                ts=0.0,
                raw={},
            )
        ]
    )
    reconciler = Reconciler(
        broker,
        limits=RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        logger=DummyLogger(),
        account_id="acct",
    )

    reconciler.reconcile([_order_request(client_order_id="abc")])

    assert broker.submitted == []


def test_reconcile_submits_when_no_matching_idempotency_key() -> None:
    broker = FakeBroker(
        open_orders=[
            OrderStatus(
                broker="fake",
                broker_order_id="open-1",
                client_order_id="abc",
                status="NEW",
                filled_qty=0.0,
                avg_price=None,
                ts=0.0,
                raw={},
            )
        ]
    )
    reconciler = Reconciler(
        broker,
        limits=RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        logger=DummyLogger(),
        account_id="acct",
    )

    intended = _order_request(client_order_id="xyz")
    reconciler.reconcile([intended])

    assert broker.submitted == [intended]


def test_reconcile_falls_back_to_client_id_when_idemp_key_missing() -> None:
    broker = FakeBroker(
        open_orders=[
            OrderStatus(
                broker="fake",
                broker_order_id="open-1",
                client_order_id="client-123",
                status="NEW",
                filled_qty=0.0,
                avg_price=None,
                ts=0.0,
                raw={},
            )
        ]
    )
    reconciler = Reconciler(
        broker,
        limits=RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        logger=DummyLogger(),
        account_id="acct",
    )

    reconciler.reconcile([_order_request(client_order_id="client-123")])

    assert broker.submitted == []


class LegacyBroker:
    def __init__(self) -> None:
        self.open_orders: list[dict[str, object]] = []
        self.positions: list[str] = []
        self.submitted: list[object] = []

    def place(self, order: object) -> dict[str, object]:
        self.submitted.append(order)
        if isinstance(order, dict):
            client_id = order.get("client_id") or order.get("client_order_id")
        else:
            client_id = getattr(order, "client_id", None) or getattr(order, "client_order_id", None)
        return {
            "status": "NEW",
            "client_id": client_id,
        }

    def fetch_open_orders(self) -> list[dict[str, object]]:
        return list(self.open_orders)

    def fetch_positions(self) -> list[str]:
        return list(self.positions)


def test_reconcile_with_legacy_broker_interfaces() -> None:
    broker = LegacyBroker()
    broker.open_orders = [
        {
            "status": "NEW",
            "client_id": "legacy-1",
        }
    ]
    logger = DummyLogger()
    reconciler = Reconciler(
        broker,
        limits=RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        logger=logger,
    )

    reconciler.reconcile([{"client_id": "legacy-1"}])

    assert broker.submitted == []

    reconciler.reconcile([{"client_id": "legacy-2"}])

    assert broker.submitted[-1]["client_id"] == "legacy-2"

    reconciler.reconcile([_order_request(client_order_id="legacy-3")])

    assert getattr(broker.submitted[-1], "client_order_id", None) == "legacy-3"

