"""Kill-switch helpers for circuit breaking trading when losses mount."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KillSwitchCfg:
    """User supplied thresholds for the :class:`KillSwitch`."""

    max_dd_pct: float
    """Maximum permitted peak-to-trough drawdown percentage."""

    max_daily_loss_pct: float
    """Maximum permitted loss versus the start of the trading day."""

    def __post_init__(self) -> None:  # pragma: no cover - tiny validation helper
        if self.max_dd_pct < 0:
            raise ValueError("max_dd_pct must be non-negative")
        if self.max_daily_loss_pct < 0:
            raise ValueError("max_daily_loss_pct must be non-negative")


class KillSwitch:
    """Track account equity and trip when configured limits are breached."""

    def __init__(self, cfg: KillSwitchCfg, start_equity: float) -> None:
        if start_equity < 0:
            raise ValueError("start_equity must be non-negative")

        self.cfg = cfg
        self.day_start_equity = float(start_equity)
        self.peak_equity = float(start_equity)

    def reset_day(self, start_equity: float) -> None:
        """Reset the day start equity when a new session begins."""

        if start_equity < 0:
            raise ValueError("start_equity must be non-negative")
        self.day_start_equity = float(start_equity)
        self.peak_equity = max(self.peak_equity, self.day_start_equity)

    def check(self, current_equity: float) -> tuple[bool, str]:
        """Check whether the kill-switch should trigger for ``current_equity``."""

        if current_equity < 0:
            raise ValueError("current_equity must be non-negative")

        self.peak_equity = max(self.peak_equity, current_equity)

        dd_pct = 0.0
        if self.peak_equity > 0:
            dd_pct = max(0.0, 100.0 * (1 - current_equity / self.peak_equity))

        daily_loss_pct = 0.0
        if self.day_start_equity > 0:
            daily_loss_pct = max(0.0, 100.0 * (1 - current_equity / self.day_start_equity))

        if dd_pct >= self.cfg.max_dd_pct:
            return True, f"Kill-switch: portfolio drawdown {dd_pct:.2f}% ≥ {self.cfg.max_dd_pct}%"

        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            return True, f"Kill-switch: daily loss {daily_loss_pct:.2f}% ≥ {self.cfg.max_daily_loss_pct}%"

        return False, ""


__all__ = ["KillSwitch", "KillSwitchCfg"]
