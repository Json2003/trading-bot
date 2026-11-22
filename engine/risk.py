"""Risk management utilities for multi-strategy coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import logging
import math

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
        limit = allocation.max_position_notional
        if limit is None:
            limit = allocation.capital
        exposures = {p.symbol: p.market_value for p in snapshot.state_for(strategy).positions}
        portfolio_exposures: dict[str, float] = {}
        for state in snapshot.states.values():
            for position in state.positions:
                portfolio_exposures[position.symbol] = portfolio_exposures.get(position.symbol, 0.0) + position.market_value

        def signed_notional(item):
            return abs(item.notional) * (1 if item.side.lower() == "buy" else -1)

        for fill in pending_fills or ():
            delta = signed_notional(fill)
            exposures[fill.symbol] = exposures.get(fill.symbol, 0.0) + delta
            portfolio_exposures[fill.symbol] = portfolio_exposures.get(fill.symbol, 0.0) + delta

        for signal in signals:
            if (signal.side.lower() not in {"buy", "sell"}
                or not math.isfinite(signal.quantity) or not math.isfinite(signal.price)
                or signal.quantity <= 0 or signal.price <= 0):
                rejected.append(signal)
                continue
            current = exposures.get(signal.symbol, 0.0)
            delta = signed_notional(signal)
            projected = current + delta
            gross = sum(abs(value) for value in exposures.values())
            projected_gross = gross - abs(current) + abs(projected)
            symbol_current = portfolio_exposures.get(signal.symbol, 0.0)
            symbol_projected = symbol_current + delta
            max_symbol = self._portfolio_limits.get("max_symbol_notional")
            # Allow reductions even when marks have already pushed the book over a limit.
            if (projected_gross > limit and projected_gross > gross) or (
                max_symbol is not None and abs(symbol_projected) > max_symbol
                and abs(symbol_projected) > abs(symbol_current)
            ):
                rejected.append(signal)
                continue
            accepted.append(signal)
            exposures[signal.symbol] = projected
            portfolio_exposures[signal.symbol] = symbol_projected

        return RiskDecision(accepted=tuple(accepted), rejected=tuple(rejected))


__all__ = ["RiskDecision", "RiskManager", "RiskViolation"]
