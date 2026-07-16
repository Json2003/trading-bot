from __future__ import annotations

import time

from tradingbot_ibkr.operator_service import TradingOperatorService
from tradingbot_ibkr.rescue_runtime import build_synthetic_paper_runtime


def test_synthetic_rescue_runtime_executes_and_replays() -> None:
    runtime = build_synthetic_paper_runtime(steps=10, seed=17)
    service = TradingOperatorService(
        broker=runtime.broker,
        orchestrator=runtime.engine,
        engine_name=runtime.name,
        cycle_interval_seconds=0.005,
    )

    service.start()
    deadline = time.monotonic() + 3.0
    while service.status().cycle_count < 12 and time.monotonic() < deadline:
        time.sleep(0.01)

    status = service.stop(cancel_open_orders=True)
    assert status.engine_configured
    assert status.engine_name == "synthetic-multi-strategy-smoke"
    assert status.cycle_count >= 12
    assert status.last_error is None
    assert not status.kill_switch_latched
    assert service.positions(), "Expected the existing strategy suite to create paper positions"
