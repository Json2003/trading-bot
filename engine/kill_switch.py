"""Portfolio level kill-switch safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from .portfolio import PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KillSwitchEvent:
    """Represents the moment a kill-switch condition was triggered."""

    timestamp: datetime
    equity: float
    reason: str


class PortfolioKillSwitch:
    """Monitors portfolio health and halts trading when limits are breached."""

    def __init__(
        self,
        *,
        max_drawdown: float | None = None,
        max_loss: float | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        if max_drawdown is not None and max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive when provided")
        if max_loss is not None and max_loss <= 0:
            raise ValueError("max_loss must be positive when provided")
        self._max_drawdown = max_drawdown
        self._max_loss = max_loss
        self._log = log or logger
        self._start_equity: float | None = None
        self._peak_equity: float | None = None
        self._strategy_peaks: dict[str, float] = {}
        self._triggered = False
        self._event: KillSwitchEvent | None = None

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def event(self) -> KillSwitchEvent | None:
        return self._event

    def reset(self) -> None:
        """Reset the kill-switch allowing trading to resume."""

        self._start_equity = None
        self._peak_equity = None
        self._strategy_peaks.clear()
        self._triggered = False
        self._event = None

    def evaluate(self, snapshot: PortfolioSnapshot, *, timestamp: datetime | None = None) -> bool:
        """Update internal state and return ``True`` when trading must stop."""

        if self._triggered:
            return True

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        equity = snapshot.total_equity
        if equity <= 0:
            self._trigger("Equity depleted", equity, timestamp)
            return True

        if self._start_equity is None:
            self._start_equity = equity
            self._peak_equity = equity
        else:
            assert self._peak_equity is not None
            if equity > self._peak_equity:
                self._peak_equity = equity

        if self._max_drawdown is not None and self._peak_equity:
            drawdown = 1 - equity / self._peak_equity if self._peak_equity else 0.0
            if drawdown >= self._max_drawdown:
                reason = f"Portfolio drawdown {drawdown:.2%} breached {self._max_drawdown:.2%}"
                self._trigger(reason, equity, timestamp)
                return True

        if self._max_loss is not None and self._start_equity is not None:
            loss = self._start_equity - equity
            if loss >= self._max_loss:
                reason = f"Portfolio loss {loss:.2f} breached limit {self._max_loss:.2f}"
                self._trigger(reason, equity, timestamp)
                return True

        for name, state in snapshot.states.items():
            limit = state.allocation.max_drawdown
            if not limit:
                continue
            equity_state = state.cash + sum(p.market_value for p in state.positions)
            peak = self._strategy_peaks.get(name)
            if peak is None or equity_state > peak:
                self._strategy_peaks[name] = equity_state
                peak = equity_state
            if peak <= 0:
                continue
            drawdown = 1 - equity_state / peak
            if drawdown >= limit:
                reason = f"Strategy {name} drawdown {drawdown:.2%} breached {limit:.2%}"
                self._trigger(reason, equity, timestamp)
                return True

        return False

    def _trigger(self, reason: str, equity: float, timestamp: datetime) -> None:
        self._triggered = True
        self._event = KillSwitchEvent(timestamp=timestamp, equity=equity, reason=reason)
        self._log.error("Kill-switch activated: %s (equity=%.2f)", reason, equity)


__all__ = ["KillSwitchEvent", "PortfolioKillSwitch"]

