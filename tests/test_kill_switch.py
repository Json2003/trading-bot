"""Unit tests for the kill-switch helpers."""

from __future__ import annotations

import pytest

from tradingbot_core.risk import KillSwitch, KillSwitchCfg


def test_drawdown_triggers_kill_switch() -> None:
    ks = KillSwitch(KillSwitchCfg(max_dd_pct=8.0, max_daily_loss_pct=9.0), start_equity=100_000.0)

    triggered, message = ks.check(91_900.0)

    assert triggered
    assert message == "Kill-switch: portfolio drawdown 8.10% ≥ 8.0%"


def test_daily_loss_triggers_kill_switch() -> None:
    ks = KillSwitch(KillSwitchCfg(max_dd_pct=10.0, max_daily_loss_pct=3.0), start_equity=100_000.0)

    triggered, message = ks.check(96_000.0)

    assert triggered
    assert message == "Kill-switch: daily loss 4.00% ≥ 3.0%"


def test_peak_updates_and_no_trigger_when_within_limits() -> None:
    ks = KillSwitch(KillSwitchCfg(max_dd_pct=12.0, max_daily_loss_pct=5.0), start_equity=100_000.0)

    triggered, message = ks.check(110_000.0)

    assert not triggered
    assert message == ""
    assert abs(ks.peak_equity - 110_000.0) < 1e-6

    triggered, message = ks.check(105_000.0)

    assert not triggered
    assert message == ""
    assert abs(ks.peak_equity - 110_000.0) < 1e-6


def test_reset_day_updates_baseline() -> None:
    ks = KillSwitch(KillSwitchCfg(max_dd_pct=100.0, max_daily_loss_pct=5.0), start_equity=100_000.0)
    ks.check(120_000.0)
    ks.reset_day(120_000.0)

    triggered, message = ks.check(110_000.0)

    assert triggered
    assert message == "Kill-switch: daily loss 8.33% ≥ 5.0%"


def test_negative_equity_inputs_raise() -> None:
    cfg = KillSwitchCfg(max_dd_pct=1.0, max_daily_loss_pct=1.0)

    with pytest.raises(ValueError):
        KillSwitch(cfg, start_equity=-1.0)

    ks = KillSwitch(cfg, start_equity=0.0)
    with pytest.raises(ValueError):
        ks.check(-1.0)

    with pytest.raises(ValueError):
        ks.reset_day(-1.0)
