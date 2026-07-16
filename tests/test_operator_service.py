from __future__ import annotations

import pytest

from tradingbot_core.strategy import OrderIntent
from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_service import TradingOperatorService


def _paper_order(broker: PaperBroker, key: str = "order-1"):
    intent = OrderIntent(
        idemp_key=key,
        symbol="AAPL",
        side="buy",
        qty=1.0,
        type="market",
    )
    return broker.submit_order(broker.intent_to_order(intent))


def test_paper_broker_deduplicates_orders_by_idempotency_key() -> None:
    broker = PaperBroker()
    first = _paper_order(broker)
    second = _paper_order(broker)

    assert first.id == second.id == "order-1"
    assert len(list(broker.list_open_orders())) == 1


def test_operator_stop_cancels_open_orders() -> None:
    broker = PaperBroker()
    _paper_order(broker)
    service = TradingOperatorService(broker=broker)

    service.start()
    status = service.stop(cancel_open_orders=True)

    assert status.state == "stopped"
    assert status.open_orders == 0


def test_emergency_stop_latches_and_blocks_restart() -> None:
    broker = PaperBroker()
    _paper_order(broker)
    service = TradingOperatorService(broker=broker)

    status = service.latch_kill_switch()

    assert status.kill_switch_latched
    assert status.open_orders == 0
    with pytest.raises(RuntimeError, match="manual recovery"):
        service.start()


def test_operator_rejects_live_mode() -> None:
    with pytest.raises(ValueError, match="paper mode only"):
        TradingOperatorService(broker=PaperBroker(), mode="live")
