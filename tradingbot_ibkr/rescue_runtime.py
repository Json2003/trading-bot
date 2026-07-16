"""Build a self-contained paper runtime for end-to-end rescue validation.

This runtime uses synthetic market data and the existing multi-strategy engine.
It is a deployment smoke environment, not evidence that any strategy is ready
for real money or that the configured crypto strategies match the planned
small-cap stock strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

from backtest.synthetic import build_synthetic_feed
from config.portfolio_loader import load_portfolio_config, load_strategy_params
from engine.builders import build_multi_strategy_orchestrator, collect_market_instruments
from execution.adapters.ccxt_executor import CCXTSignalExecutor
from tradingbot_ibkr.execution.paper_broker import PaperBroker

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


class SyntheticPaperEngine:
    """Replay synthetic data continuously through the existing engine."""

    def __init__(self, orchestrator: Any, data_feed: Any) -> None:
        self._orchestrator = orchestrator
        self._data_feed = data_feed

    def step(self) -> Any:
        try:
            return self._orchestrator.run_cycle()
        except StopIteration:
            self._data_feed.reset()
            return self._orchestrator.run_cycle()

    @property
    def kill_switch_triggered(self) -> bool:
        return bool(getattr(self._orchestrator, "kill_switch_triggered", False))

    @property
    def kill_switch_event(self) -> Any:
        return getattr(self._orchestrator, "kill_switch_event", None)


@dataclass(frozen=True, slots=True)
class RescueRuntime:
    broker: PaperBroker
    engine: SyntheticPaperEngine
    portfolio: Any
    name: str = "synthetic-multi-strategy-smoke"


def build_synthetic_paper_runtime(
    *,
    steps: int = 2_000,
    seed: int = 11,
    fees_bps: float = 5.0,
    slippage_bps: float = 2.0,
    portfolio_config: str | Path | None = None,
    strategy_dir: str | Path | None = None,
) -> RescueRuntime:
    """Create a complete paper broker, data feed and deterministic cycle engine."""

    if steps < 10:
        raise ValueError("synthetic runtime requires at least 10 steps")

    portfolio_path = Path(portfolio_config or REPO_ROOT / "config" / "portfolio.yaml")
    strategy_path = Path(strategy_dir or REPO_ROOT / "config" / "strategy")
    portfolio_cfg = load_portfolio_config(str(portfolio_path))
    strategy_params = {
        cfg.name: load_strategy_params(cfg.name, str(strategy_path))
        for cfg in portfolio_cfg.strategies
    }
    instruments = collect_market_instruments(strategy_params, default_timeframe="1h")
    data_feed, _ = build_synthetic_feed(
        instruments,
        steps=steps,
        seed=seed,
        timeframe="1h",
    )

    broker = PaperBroker()
    executor = CCXTSignalExecutor(
        broker,
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
        log=LOGGER,
    )
    orchestrator, portfolio, _, _, _ = build_multi_strategy_orchestrator(
        portfolio_config=portfolio_cfg,
        strategy_params=strategy_params,
        clients={},
        broker=broker,
        executor=executor,
        default_timeframe="1h",
        ohlcv_candles=120,
        log=LOGGER,
        data_feed=data_feed,
    )
    engine = SyntheticPaperEngine(orchestrator, data_feed)
    return RescueRuntime(broker=broker, engine=engine, portfolio=portfolio)


__all__ = ["RescueRuntime", "SyntheticPaperEngine", "build_synthetic_paper_runtime"]
