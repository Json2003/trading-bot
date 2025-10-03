from __future__ import annotations

from typing import Iterable, List

from tradingbot_ibkr.execution.broker_base import Order, Position
from tradingbot_ibkr.execution.reconciler import Reconciler


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
        local_orders={"1": Order(id="1", symbol="BTC", side="buy", quantity=1.0, filled_quantity=0.4)},
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
