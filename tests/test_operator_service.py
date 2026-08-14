from __future__ import annotations

import time
from threading import Event

import pytest

from tradingbot_core.strategy import OrderIntent
from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_service import TradingOperatorService


class CountingEngine:
    def __init__(self) -> None:
        self.cycles = 0
        self.cycled = Event()

    def step(self) -> None:
        self.cycles += 1
        self.cycled.set()


class FailingEngine:
    def step(self) -> None:
        raise ValueError("feed disconnected")


def _paper_order(broker: PaperBroker, key: str = "order-1"):
    intent = OrderIntent(
        idemp_key=key,
        symbol="AAPL",
        side="buy",
        qty=1.0,
        type="market",
    )
    return broker.submit_order(broker.intent_to_order(intent))


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_paper_broker_deduplicates_orders_by_idempotency_key() -> None:
    broker = PaperBroker()
    first = _paper_order(broker)
    second = _paper_order(broker)

    assert first.id == second.id == "order-1"
    assert len(list(broker.list_open_orders())) == 1


def test_operator_refuses_to_report_running_without_engine() -> None:
    service = TradingOperatorService(broker=PaperBroker())

    with pytest.raises(RuntimeError, match="no trading engine"):
        service.start()

    status = service.status()
    assert status.state == "stopped"
    assert status.engine_configured is False
    assert status.cycle_count == 0


def test_operator_runs_engine_and_stop_cancels_open_orders() -> None:
    broker = PaperBroker()
    engine = CountingEngine()
    _paper_order(broker)
    service = TradingOperatorService(
        broker=broker,
        orchestrator=engine,
        cycle_interval_seconds=0.02,
    )

    started = service.start()
    assert started.state == "running"
    assert engine.cycled.wait(timeout=1.0)
    assert _wait_for(lambda: service.status().cycle_count >= 1)

    status = service.stop(cancel_open_orders=True)

    assert status.state == "stopped"
    assert status.open_orders == 0
    assert status.cycle_count >= 1
    assert status.last_cycle_at is not None


def test_operator_pause_stops_new_cycles_until_resumed() -> None:
    engine = CountingEngine()
    service = TradingOperatorService(
        broker=PaperBroker(),
        orchestrator=engine,
        cycle_interval_seconds=0.05,
    )

    service.start()
    assert engine.cycled.wait(timeout=1.0)
    paused = service.pause()
    assert paused.state == "paused"
    count_after_pause = service.status().cycle_count
    time.sleep(0.12)
    assert service.status().cycle_count == count_after_pause

    resumed = service.start()
    assert resumed.state == "running"
    assert _wait_for(lambda: service.status().cycle_count > count_after_pause)
    service.stop()


def test_engine_fault_latches_stop_and_cancels_orders() -> None:
    broker = PaperBroker()
    _paper_order(broker)
    service = TradingOperatorService(
        broker=broker,
        orchestrator=FailingEngine(),
        cycle_interval_seconds=0.01,
    )

    service.start()
    assert _wait_for(lambda: service.status().state == "faulted")
    status = service.status()

    assert status.kill_switch_latched
    assert status.open_orders == 0
    assert "feed disconnected" in (status.last_error or "")
    with pytest.raises(RuntimeError, match="manual recovery"):
        service.start()


def test_emergency_stop_latches_and_blocks_restart() -> None:
    broker = PaperBroker()
    engine = CountingEngine()
    _paper_order(broker)
    service = TradingOperatorService(broker=broker, orchestrator=engine)

    status = service.latch_kill_switch()

    assert status.kill_switch_latched
    assert status.open_orders == 0
    with pytest.raises(RuntimeError, match="manual recovery"):
        service.start()


def test_operator_rejects_live_mode() -> None:
    with pytest.raises(ValueError, match="paper mode only"):
        TradingOperatorService(broker=PaperBroker(), mode="live")
