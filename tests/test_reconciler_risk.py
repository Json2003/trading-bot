"""Tests for the reconciliation risk helpers."""

from __future__ import annotations

import logging

import pytest

from tradingbot_ibkr.execution import PaperBroker, Reconciler, RiskEvaluation, RiskLimits


def test_risk_limits_validate_inputs() -> None:
    with pytest.raises(ValueError):
        RiskLimits(max_daily_loss_pct=-1.0, kill_switch_drawdown_pct=5.0, max_position_risk_pct=1.0)


def test_evaluate_risk_triggers_breaches(caplog: pytest.LogCaptureFixture) -> None:
    broker = PaperBroker()
    limits = RiskLimits(max_daily_loss_pct=3.0, kill_switch_drawdown_pct=8.0, max_position_risk_pct=1.0)
    reconciler = Reconciler(broker, limits=limits, logger=logging.getLogger("test.reconciler"))

    with caplog.at_level(logging.WARNING, logger="test.reconciler"):
        evaluation = reconciler.evaluate_risk(
            daily_loss_pct=4.0,
            drawdown_pct=2.0,
            position_risk_pct=0.5,
        )

    assert isinstance(evaluation, RiskEvaluation)
    assert evaluation.breached_limits == ("max_daily_loss_pct",)
    assert evaluation.kill_switch_triggered
    assert "Risk limits breached" in caplog.text


def test_evaluate_risk_without_limits() -> None:
    reconciler = Reconciler(PaperBroker())

    evaluation = reconciler.evaluate_risk(
        daily_loss_pct=10.0,
        drawdown_pct=10.0,
        position_risk_pct=2.0,
    )

    assert evaluation.breached_limits == ()
    assert not evaluation.kill_switch_triggered
