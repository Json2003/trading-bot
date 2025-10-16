"""Risk management configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class RiskConfigError(ValueError):
    """Raised when an invalid risk configuration is supplied."""


@dataclass(frozen=True)
class RiskCfg:
    """Configuration for portfolio level risk controls.

    Parameters are expressed in human readable percentages rather than
    fractions (e.g. ``1.0`` represents one percent).
    """

    per_trade_risk_pct: float
    max_daily_loss_pct: float
    kill_switch_drawdown_pct: float
    max_leverage: float

    def __post_init__(self) -> None:  # pragma: no cover - exercised via ``from_mapping``
        per_trade = float(self.per_trade_risk_pct)
        max_daily = float(self.max_daily_loss_pct)
        kill_switch = float(self.kill_switch_drawdown_pct)
        leverage = float(self.max_leverage)

        if not 0.5 <= per_trade <= 2.0:
            raise RiskConfigError(
                "per_trade_risk_pct must be between 0.5% and 2.0% inclusive."
            )

        if max_daily <= 0:
            raise RiskConfigError("max_daily_loss_pct must be positive.")

        if max_daily < per_trade:
            raise RiskConfigError(
                "max_daily_loss_pct must be greater than or equal to per_trade_risk_pct."
            )

        if kill_switch <= max_daily:
            raise RiskConfigError(
                "kill_switch_drawdown_pct must exceed max_daily_loss_pct to act as an "
                "emergency circuit breaker."
            )

        if leverage < 1.0:
            raise RiskConfigError("max_leverage must be at least 1.0x.")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RiskCfg":
        """Create an instance from a generic mapping.

        Values are coerced to ``float`` and validated. Missing keys raise a
        ``KeyError`` so callers receive a clear signal that required
        configuration is absent.
        """

        required_keys = {
            "per_trade_risk_pct",
            "max_daily_loss_pct",
            "kill_switch_drawdown_pct",
            "max_leverage",
        }

        missing = sorted(required_keys.difference(data))
        if missing:
            raise KeyError(f"Risk configuration missing keys: {', '.join(missing)}")

        try:
            per_trade = float(data["per_trade_risk_pct"])
            max_daily = float(data["max_daily_loss_pct"])
            kill_switch = float(data["kill_switch_drawdown_pct"])
            leverage = float(data["max_leverage"])
        except (TypeError, ValueError) as exc:  # pragma: no cover - construction enforces type
            raise RiskConfigError("Risk configuration values must be numeric.") from exc

        return cls(
            per_trade_risk_pct=per_trade,
            max_daily_loss_pct=max_daily,
            kill_switch_drawdown_pct=kill_switch,
            max_leverage=leverage,
        )

    def as_dict(self) -> dict[str, float]:
        """Return the configuration as a plain dictionary for serialisation."""

        return {
            "per_trade_risk_pct": float(self.per_trade_risk_pct),
            "max_daily_loss_pct": float(self.max_daily_loss_pct),
            "kill_switch_drawdown_pct": float(self.kill_switch_drawdown_pct),
            "max_leverage": float(self.max_leverage),
        }


__all__ = ["RiskCfg", "RiskConfigError"]

