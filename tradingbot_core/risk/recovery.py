"""Guarded recovery policy for paper-only kill-switch incidents.

The policy deliberately separates a one-time paper re-arm from a full reset.
An emergency stop, engine fault, or broker reconciliation failure is never
auto-reset. A drawdown stop can be re-armed only after the account is flat,
healthy, below a much smaller drawdown band, and stable for consecutive
observations. Clearing the historical peak/day baseline still requires an
explicit human approval.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RecoveryCfg:
    """Conservative recovery thresholds expressed as fractions."""

    kill_switch_drawdown_fraction: float = 0.02
    rearm_drawdown_fraction: float = 0.005
    max_recovery_volatility_fraction: float = 0.04
    stable_cycles: int = 12
    full_reset_stable_cycles: int = 72
    max_rearm_attempts: int = 1
    require_flat: bool = True
    full_reset_requires_human_approval: bool = True

    def __post_init__(self) -> None:
        values = (
            self.kill_switch_drawdown_fraction,
            self.rearm_drawdown_fraction,
            self.max_recovery_volatility_fraction,
        )
        if any(not math.isfinite(float(value)) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("recovery fractions must be between 0 and 1")
        if self.rearm_drawdown_fraction >= self.kill_switch_drawdown_fraction:
            raise ValueError("rearm drawdown must be below the kill-switch drawdown")
        if self.stable_cycles < 1:
            raise ValueError("stable_cycles must be positive")
        if self.full_reset_stable_cycles < self.stable_cycles:
            raise ValueError("full reset stability must be at least rearm stability")
        if self.max_rearm_attempts < 1:
            raise ValueError("max_rearm_attempts must be positive")


@dataclass(frozen=True, slots=True)
class RecoveryStatus:
    """Serializable state exposed to a paper operator."""

    state: str
    reason: str | None
    stable_cycles: int
    post_rearm_stable_cycles: int
    rearm_attempts: int
    can_auto_rearm: bool
    can_full_reset: bool


class PaperRecoveryController:
    """Track whether a paper drawdown stop has earned a safe re-arm."""

    def __init__(self, cfg: RecoveryCfg | None = None) -> None:
        self.cfg = cfg or RecoveryCfg()
        self._state = "armed"
        self._reason: str | None = None
        self._recoverable = False
        self._stable_cycles = 0
        self._post_rearm_stable_cycles = 0
        self._rearm_attempts = 0
        self._eligible = False

    @property
    def state(self) -> str:
        return self._state

    def trip(self, reason: str, *, manual: bool = False, recoverable: bool = True) -> None:
        """Latch the controller after an incident.

        ``manual`` and non-recoverable incidents cannot be auto-rearmed. A
        drawdown monitor may call this with ``recoverable=True``.
        """

        self._state = "manual_latched" if manual else "latched"
        self._reason = str(reason)
        self._recoverable = bool(recoverable and not manual)
        self._stable_cycles = 0
        self._post_rearm_stable_cycles = 0
        self._eligible = False

    def observe(
        self,
        *,
        current_drawdown_fraction: float,
        realized_volatility_fraction: float,
        engine_healthy: bool,
        open_orders: int,
        open_positions: int,
    ) -> RecoveryStatus:
        """Record one trusted monitor observation and update stability."""

        self._validate_observation(
            current_drawdown_fraction,
            realized_volatility_fraction,
            open_orders,
            open_positions,
        )
        flat = open_orders == 0 and open_positions == 0
        stable = (
            self._recoverable
            and bool(engine_healthy)
            and current_drawdown_fraction <= self.cfg.rearm_drawdown_fraction
            and realized_volatility_fraction <= self.cfg.max_recovery_volatility_fraction
            and (flat or not self.cfg.require_flat)
        )

        if self._state == "latched":
            self._stable_cycles = self._stable_cycles + 1 if stable else 0
            self._eligible = self._stable_cycles >= self.cfg.stable_cycles
        elif self._state == "rearmed":
            self._post_rearm_stable_cycles = (
                self._post_rearm_stable_cycles + 1 if stable else 0
            )
            self._eligible = False
        else:
            self._eligible = False

        return self.status()

    def can_auto_rearm(self) -> bool:
        return bool(
            self._state == "latched"
            and self._recoverable
            and self._eligible
            and self._rearm_attempts < self.cfg.max_rearm_attempts
        )

    def auto_rearm(self) -> RecoveryStatus:
        """Consume the one-time re-arm after ``observe`` proves stability."""

        if not self.can_auto_rearm():
            raise RuntimeError("paper recovery stability gate has not passed")
        self._state = "rearmed"
        self._rearm_attempts += 1
        self._eligible = False
        return self.status()

    def can_full_reset(self) -> bool:
        return bool(
            self._state == "rearmed"
            and self._post_rearm_stable_cycles >= self.cfg.full_reset_stable_cycles
        )

    def full_reset(self, *, human_approved: bool = False) -> RecoveryStatus:
        """Clear incident history only after the post-rearm stability window."""

        if not self.can_full_reset():
            raise RuntimeError("full reset requires the post-rearm stability window")
        if self.cfg.full_reset_requires_human_approval and not human_approved:
            raise PermissionError("full reset requires explicit human approval")
        self._state = "armed"
        self._reason = None
        self._recoverable = False
        self._stable_cycles = 0
        self._post_rearm_stable_cycles = 0
        self._rearm_attempts = 0
        self._eligible = False
        return self.status()

    def status(self) -> RecoveryStatus:
        return RecoveryStatus(
            state=self._state,
            reason=self._reason,
            stable_cycles=self._stable_cycles,
            post_rearm_stable_cycles=self._post_rearm_stable_cycles,
            rearm_attempts=self._rearm_attempts,
            can_auto_rearm=self.can_auto_rearm(),
            can_full_reset=self.can_full_reset(),
        )

    @staticmethod
    def _validate_observation(
        current_drawdown_fraction: float,
        realized_volatility_fraction: float,
        open_orders: int,
        open_positions: int,
    ) -> None:
        if not math.isfinite(float(current_drawdown_fraction)) or not 0.0 <= float(
            current_drawdown_fraction
        ) <= 1.0:
            raise ValueError("current_drawdown_fraction must be between 0 and 1")
        if not math.isfinite(float(realized_volatility_fraction)) or not 0.0 <= float(
            realized_volatility_fraction
        ) <= 1.0:
            raise ValueError("realized_volatility_fraction must be between 0 and 1")
        if int(open_orders) < 0 or int(open_positions) < 0:
            raise ValueError("open order and position counts cannot be negative")


__all__ = ["PaperRecoveryController", "RecoveryCfg", "RecoveryStatus"]
