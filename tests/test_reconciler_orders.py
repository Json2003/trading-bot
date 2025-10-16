from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from tradingbot_ibkr.execution.broker_base import Order, Position
from tradingbot_ibkr.execution.reconciler import Reconciler, RiskLimits


class StaticBroker:
    def __init__(self, orders: Iterable[Order], positions: Iterable[Position]):
        self._orders = list(orders)
        self._positions = list(positions)

    def list_open_orders(self) -> List[Order]:
        return list(self._orders)

    def list_positions(self) -> List[Position]:
        return list(self._positions)


def test_reconcile_identifies_partial_fills() -> None:
    broker = StaticBroker(
        [Order(id="1", symbol="BTC", side="buy", quantity=1.0, filled_quantity=0.4)],
        [],
    )
    reconciler = Reconciler(broker)

    report = reconciler.reconcile(
        local_orders={
            "1": Order(id="1", symbol="BTC", side="buy", quantity=1.0, filled_quantity=0.4)
        },
        local_positions={},
    )

    assert report.partially_filled_orders == ("1",)


class FlakyBroker(StaticBroker):
    def __init__(self) -> None:
        super().__init__(orders=[], positions=[])
        self._call = 0

    def list_open_orders(self) -> List[Order]:
        self._call += 1
        if self._call == 1:
            return [Order(id="retry", symbol="ETH", side="buy", quantity=2.0, filled_quantity=0.0)]
        return []


def test_reconcile_with_retry_resolves_mismatch() -> None:
    broker = FlakyBroker()
    reconciler = Reconciler(broker)
    sleeps: list[float] = []

    report = reconciler.reconcile_with_retry(
        local_orders={},
        local_positions={},
        attempts=3,
        backoff=0.25,
        sleeper=sleeps.append,
    )

    assert report.is_clean
    assert sleeps == [0.25]


class RecordingBroker(StaticBroker):
    def __init__(self) -> None:
        super().__init__(orders=[], positions=[])
        self.submitted: List[Order] = []

    def submit_order(self, order: Order) -> Order:
        recorded = replace(order)
        self.submitted.append(recorded)
        self._orders.append(recorded)
        return recorded


def test_submit_idempotent_reuses_existing_order() -> None:
    existing = Order(id="abc", symbol="BTC", side="buy", quantity=1.0)
    broker = RecordingBroker()
    broker._orders.append(existing)
    reconciler = Reconciler(broker)

    result = reconciler.submit_idempotent(existing)

    assert result is existing
    assert broker.submitted == []


def test_submit_idempotent_places_missing_order() -> None:
    broker = RecordingBroker()
    reconciler = Reconciler(broker)
    order = Order(id="new", symbol="ETH", side="sell", quantity=0.5)

    result = reconciler.submit_idempotent(order)

    assert result.id == order.id
    assert broker.submitted and broker.submitted[0].id == order.id


def test_check_kill_switch_triggers_on_limits() -> None:
    broker = StaticBroker(orders=[], positions=[])
    limits = RiskLimits(
        max_daily_loss_pct=5.0,
        kill_switch_drawdown_pct=10.0,
        max_position_risk_pct=100.0,
    )
    reconciler = Reconciler(broker, limits=limits)

    triggered = reconciler.check_kill_switch([100_000.0, 105_000.0, 90_000.0])

    assert triggered


def test_check_kill_switch_ignores_when_limits_missing() -> None:
    broker = StaticBroker(orders=[], positions=[])
    reconciler = Reconciler(broker)

    assert not reconciler.check_kill_switch([100.0, 95.0])
