"""High level orchestration for coordinating multiple strategies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, Sequence
import logging

from .datafeed import MarketData, UnifiedDataFeed
from .kill_switch import PortfolioKillSwitch
from .portfolio import OrderFill, Portfolio, PortfolioSnapshot, StrategyAllocation
from .position_sizing import ATRSizingConfig, atr_position_size, atr_stop
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
    sizing: ATRSizingConfig | None = None


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
        kill_switch: PortfolioKillSwitch | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy must be configured")
        self._data_feed = data_feed
        self._portfolio = portfolio
        self._risk = risk_manager
        self._strategies = list(strategies)
        self._executor = executor
        self._kill_switch = kill_switch
        self._log = log or logger

    def run_cycle(self) -> list[OrderFill]:
        """Run a single evaluation cycle across all strategies."""

        market_data = self._data_feed.fetch()
        snapshot = self._portfolio.snapshot(
            mark_prices={data.symbol: data.price for data in market_data.values()}
        )
        timestamp = datetime.now(timezone.utc)
        if self._kill_switch and self._kill_switch.evaluate(snapshot, timestamp=timestamp):
            event = self._kill_switch.event
            if event:
                self._log.warning("Kill-switch active (%s) – skipping cycle", event.reason)
            else:
                self._log.warning("Kill-switch active – skipping cycle")
            return []
        fills: list[OrderFill] = []

        for binding in self._strategies:
            context = self._build_context(binding, market_data, snapshot, timestamp)
            signals = list(binding.strategy.generate_signals(context))
            if not signals:
                continue

            sized_signals: list[StrategySignal] = []
            for signal in signals:
                sized = self._apply_sizing(
                    binding, signal, market_data, snapshot
                )
                if sized is not None:
                    sized_signals.append(sized)

            if not sized_signals:
                continue

            decision = self._risk.evaluate_signals(
                binding.name, sized_signals, snapshot
            )
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

    def _apply_sizing(
        self,
        binding: StrategyBinding,
        signal: StrategySignal,
        market_data: dict[str, MarketData],
        snapshot: PortfolioSnapshot,
    ) -> StrategySignal | None:
        config = binding.sizing
        if config is None:
            return signal

        market_key = str(signal.tags.get("market_key", signal.symbol))
        market = market_data.get(market_key)
        if market is None:
            self._log.warning(
                "Skipping sizing for %s: missing market data for %s",
                binding.name,
                market_key,
            )
            return None

        state = snapshot.state_for(binding.name)
        equity = state.cash + sum(position.market_value for position in state.positions)

        sizing = atr_position_size(
            equity,
            market,
            config=config,
            price=signal.price,
        )
        if not sizing.is_actionable:
            self._log.debug(
                "ATR sizing filtered %s signal for %s (equity=%.2f)",
                binding.name,
                market_key,
                equity,
            )
            return None

        tags = dict(signal.tags)
        tags["atr"] = {
            "value": sizing.atr,
            "risk_cash": sizing.risk_cash,
            "stop_distance": sizing.stop_distance,
        }
        stop = sizing.stop_distance or 0.0
        tags["stop_level"] = atr_stop(signal.price, stop, signal.side)

        return replace(signal, quantity=sizing.quantity, tags=tags)


__all__ = ["OrderExecutor", "StrategyBinding", "MultiStrategyOrchestrator"]
