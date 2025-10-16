"""High level orchestration for coordinating multiple strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence
import logging

from .datafeed import MarketData, UnifiedDataFeed
from .portfolio import OrderFill, Portfolio, PortfolioSnapshot, StrategyAllocation
from .risk import RiskManager
from strategies.base import Strategy, StrategyContext, StrategySignal

logger = logging.getLogger(__name__)


class OrderExecutor(Protocol):
    """Protocol for submitting strategy signals to an execution venue."""

    def execute(self, signal: StrategySignal) -> OrderFill | None:
        ...


@dataclass(frozen=True)
class StrategyBinding:
    """Glue object combining a strategy implementation with its config."""

    name: str
    strategy: Strategy
    allocation: StrategyAllocation


class MultiStrategyOrchestrator:
    """Coordinates market data, strategies, risk checks and execution."""

    def __init__(
        self,
        *,
        data_feed: UnifiedDataFeed,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        strategies: Sequence[StrategyBinding],
        executor: OrderExecutor | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy must be configured")
        self._data_feed = data_feed
        self._portfolio = portfolio
        self._risk = risk_manager
        self._strategies = list(strategies)
        self._executor = executor
        self._log = log or logger

    def run_cycle(self) -> list[OrderFill]:
        """Run a single evaluation cycle across all strategies."""

        market_data = self._data_feed.fetch()
        snapshot = self._portfolio.snapshot(
            mark_prices={data.symbol: data.price for data in market_data.values()}
        )
        timestamp = datetime.now(timezone.utc)
        fills: list[OrderFill] = []

        for binding in self._strategies:
            context = self._build_context(binding, market_data, snapshot, timestamp)
            signals = list(binding.strategy.generate_signals(context))
            if not signals:
                continue

            decision = self._risk.evaluate_signals(binding.name, signals, snapshot)
            if not decision.accepted:
                self._log.debug("No signals accepted for %s", binding.name)
                continue

            if self._executor is None:
                self._log.debug("Executor not configured; skipping execution for %s", binding.name)
                continue

            for signal in decision.accepted:
                fill = self._executor.execute(signal)
                if fill:
                    fills.append(fill)
                    self._portfolio.apply_fills(binding.name, [fill])

        return fills

    def _build_context(
        self,
        binding: StrategyBinding,
        market_data: dict[str, MarketData],
        snapshot: PortfolioSnapshot,
        timestamp: datetime,
    ) -> StrategyContext:
        state = snapshot.state_for(binding.name)
        return StrategyContext(
            strategy=binding.name,
            timestamp=timestamp,
            market_data=market_data,
            allocation=binding.allocation,
            cash=state.cash,
            positions=state.positions,
            pnl=state.pnl,
        )


__all__ = ["OrderExecutor", "StrategyBinding", "MultiStrategyOrchestrator"]
