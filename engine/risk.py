"""Risk management utilities for multi-strategy coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import logging

from .portfolio import OrderFill, PortfolioSnapshot, StrategyAllocation
from strategies.base import StrategySignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of evaluating a strategy's proposed signals."""

    accepted: tuple[StrategySignal, ...]
    rejected: tuple[StrategySignal, ...]


class RiskViolation(RuntimeError):
    """Raised when a signal violates hard risk constraints."""

    def __init__(self, message: str, *, signal: StrategySignal | None = None) -> None:
        super().__init__(message)
        self.signal = signal


class RiskManager:
    """Performs per-strategy and portfolio wide risk checks."""

    def __init__(
        self,
        limits: Mapping[str, StrategyAllocation],
        *,
        portfolio_limits: Mapping[str, float] | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._limits = dict(limits)
        self._portfolio_limits = dict(portfolio_limits or {})
        self._log = log or logger

    def evaluate_signals(
        self,
        strategy: str,
        signals: Sequence[StrategySignal],
        snapshot: PortfolioSnapshot,
        *,
        pending_fills: Iterable[OrderFill] | None = None,
    ) -> RiskDecision:
        allocation = self._limits[strategy]
        accepted: list[StrategySignal] = []
        rejected: list[StrategySignal] = []
        available_notional = allocation.max_position_notional or allocation.capital

        if pending_fills:
            for fill in pending_fills:
                notional = fill.notional
                available_notional -= notional

        if available_notional <= 0:
            self._log.warning("Strategy %s has no remaining notional capacity", strategy)
            return RiskDecision(accepted=(), rejected=tuple(signals))

        for signal in signals:
            notional = abs(signal.notional)
            if notional > available_notional:
                self._log.info(
                    "Rejecting %s signal for %s: notional %.2f exceeds remaining capacity %.2f",
                    strategy,
                    signal.symbol,
                    notional,
                    available_notional,
                )
                rejected.append(signal)
                continue

            if self._violates_portfolio_limits(signal, snapshot):
                rejected.append(signal)
                continue

            accepted.append(signal)
            available_notional -= notional

        return RiskDecision(accepted=tuple(accepted), rejected=tuple(rejected))

    def _violates_portfolio_limits(self, signal: StrategySignal, snapshot: PortfolioSnapshot) -> bool:
        max_symbol = self._portfolio_limits.get("max_symbol_notional")
        if max_symbol is None:
            return False
        state = snapshot.state_for(signal.strategy)
        current = sum(p.market_value for p in state.positions if p.symbol == signal.symbol)
        projected = current + signal.notional
        if abs(projected) > max_symbol:
            self._log.info(
                "Rejecting signal for %s: projected exposure %.2f exceeds symbol limit %.2f",
                signal.symbol,
                projected,
                max_symbol,
            )
            return True
        return False


__all__ = ["RiskDecision", "RiskManager", "RiskViolation"]
