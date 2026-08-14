from __future__ import annotations

import pytest

from tradingbot_core.risk import PaperRecoveryController, RecoveryCfg
from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_service import TradingOperatorService


def _stable_observation(controller: PaperRecoveryController) -> None:
    controller.observe(
        current_drawdown_fraction=0.001,
        realized_volatility_fraction=0.01,
        engine_healthy=True,
        open_orders=0,
        open_positions=0,
    )


def test_recovery_requires_flat_stability_and_human_full_reset() -> None:
    controller = PaperRecoveryController(
        RecoveryCfg(stable_cycles=2, full_reset_stable_cycles=3, max_rearm_attempts=1)
    )
    controller.trip("drawdown", recoverable=True)

    controller.observe(
        current_drawdown_fraction=0.001,
        realized_volatility_fraction=0.01,
        engine_healthy=True,
        open_orders=1,
        open_positions=0,
    )
    assert not controller.can_auto_rearm()
    _stable_observation(controller)
    assert not controller.can_auto_rearm()
    _stable_observation(controller)
    controller.auto_rearm()

    for _ in range(2):
        _stable_observation(controller)
    assert not controller.can_full_reset()
    _stable_observation(controller)
    assert controller.can_full_reset()
    with pytest.raises(PermissionError):
        controller.full_reset()
    assert controller.full_reset(human_approved=True).state == "armed"


def test_manual_stop_never_auto_rearms() -> None:
    controller = PaperRecoveryController(RecoveryCfg(stable_cycles=1))
    controller.trip("manual emergency stop", manual=True)
    _stable_observation(controller)
    assert controller.status().state == "manual_latched"
    assert not controller.can_auto_rearm()


def test_operator_can_only_auto_rearm_recoverable_paper_drawdown() -> None:
    service = TradingOperatorService(
        broker=PaperBroker(),
        recovery_config=RecoveryCfg(stable_cycles=2, full_reset_stable_cycles=2),
    )
    service.latch_kill_switch(reason="2% drawdown", recoverable=True)
    assert service.status().kill_switch_latched

    service.evaluate_recovery(
        current_drawdown_fraction=0.001,
        realized_volatility_fraction=0.01,
        engine_healthy=True,
    )
    service.evaluate_recovery(
        current_drawdown_fraction=0.001,
        realized_volatility_fraction=0.01,
        engine_healthy=True,
    )
    status = service.status()
    assert not status.kill_switch_latched
    assert status.recovery_state == "rearmed"

