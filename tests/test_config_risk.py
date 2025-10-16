"""Unit tests for the risk configuration dataclass."""

from __future__ import annotations

import pytest

from tradingbot_core.config.risk import RiskCfg, RiskConfigError


def test_from_mapping_happy_path() -> None:
    cfg = RiskCfg.from_mapping(
        {
            "per_trade_risk_pct": "1.0",
            "max_daily_loss_pct": 3,
            "kill_switch_drawdown_pct": 8,
            "max_leverage": "5",
        }
    )

    assert cfg.as_dict() == {
        "per_trade_risk_pct": 1.0,
        "max_daily_loss_pct": 3.0,
        "kill_switch_drawdown_pct": 8.0,
        "max_leverage": 5.0,
    }


def test_from_mapping_missing_keys() -> None:
    with pytest.raises(KeyError):
        RiskCfg.from_mapping({"per_trade_risk_pct": 1.0})


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("per_trade_risk_pct", 0.1, "between 0.5% and 2.0%"),
        ("per_trade_risk_pct", 3.0, "between 0.5% and 2.0%"),
        ("max_daily_loss_pct", -1.0, "must be positive"),
        ("max_daily_loss_pct", 0.25, "greater than or equal"),
        ("kill_switch_drawdown_pct", 2.0, "must exceed"),
        ("max_leverage", 0.5, "at least 1.0x"),
    ],
)
def test_validation_errors(field: str, value: float, message: str) -> None:
    base = {
        "per_trade_risk_pct": 1.0,
        "max_daily_loss_pct": 3.0,
        "kill_switch_drawdown_pct": 8.0,
        "max_leverage": 5.0,
    }
    base[field] = value

    with pytest.raises(RiskConfigError) as exc:
        RiskCfg.from_mapping(base)

    assert message in str(exc.value)

