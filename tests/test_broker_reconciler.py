from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from brokers.reconciler import Reconciler, RiskLimits


@dataclass
class FakeOrder:
    client_id: str | None
    idemp_key: str | None


@dataclass
class FakeStatus:
    status: str
    client_id: str | None = None


@dataclass
class FakeBroker:
    open_orders: List[FakeOrder] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    submitted: List[FakeOrder] = field(default_factory=list)

    def place(self, order: FakeOrder) -> FakeStatus:
        self.submitted.append(order)
        return FakeStatus(status="submitted", client_id=order.client_id)

    def fetch_open_orders(self) -> Iterable[FakeOrder]:
        return list(self.open_orders)

    def fetch_positions(self) -> Iterable[str]:
        return list(self.positions)


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[object, ...]]] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append((message, args))

    def error(self, message: str, *args: object) -> None:
        self.messages.append((message, args))


def test_reconcile_ignores_orders_with_same_idempotency_key() -> None:
    broker = FakeBroker(open_orders=[FakeOrder(client_id="open-1", idemp_key="abc")])
    reconciler = Reconciler(
        broker,
        RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        DummyLogger(),
    )

    reconciler.reconcile([FakeOrder(client_id="intended-1", idemp_key="abc")])

    assert broker.submitted == []


def test_reconcile_submits_when_no_matching_idempotency_key() -> None:
    broker = FakeBroker(open_orders=[FakeOrder(client_id="open-1", idemp_key="abc")])
    reconciler = Reconciler(
        broker,
        RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        DummyLogger(),
    )

    reconciler.reconcile([FakeOrder(client_id="intended-1", idemp_key="xyz")])

    assert broker.submitted == [FakeOrder(client_id="intended-1", idemp_key="xyz")]


def test_reconcile_falls_back_to_client_id_when_idemp_key_missing() -> None:
    broker = FakeBroker(open_orders=[FakeOrder(client_id="client-123", idemp_key=None)])
    reconciler = Reconciler(
        broker,
        RiskLimits(max_daily_loss_pct=10, kill_switch_drawdown_pct=15, max_position_risk_pct=5),
        DummyLogger(),
    )

    reconciler.reconcile([FakeOrder(client_id="client-123", idemp_key=None)])

    assert broker.submitted == []

