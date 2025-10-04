"""Utilities that orchestrate running a backtest end-to-end.

The harness defined here is intentionally lightweight so that strategies and
research notebooks can share a consistent code path when producing result
artifacts.  It handles bookkeeping such as capturing the git revision, storing
configured broker fees and random seeds and computing a small set of portfolio
metrics that most of our dashboards expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping
import os
import subprocess
import time

from .backtest_save import save_backtest_results
from .metrics import PortfolioStats, compute_portfolio_stats


def _detect_git_sha() -> str:
    """Return the git SHA for the current repository if available."""

    env_sha = os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "<unknown>"
    return sha or "<unknown>"


@dataclass(slots=True)
class BacktestContext:
    """Configuration captured for a single backtest run."""

    strategy: str
    market: str
    timeframe: str
    seed: int | None = None
    broker_fees: Mapping[str, float] | None = None
    parameters: Mapping[str, Any] | None = None
    tags: Iterable[str] | None = None

    def as_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "strategy": self.strategy,
            "market": self.market,
            "timeframe": self.timeframe,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.broker_fees is not None:
            payload["fees"] = dict(self.broker_fees)
        if self.parameters is not None:
            payload["parameters"] = dict(self.parameters)
        if self.tags is not None:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(slots=True)
class BacktestResult:
    """Container representing the outcome of a backtest."""

    equity_curve: Iterable[float]
    returns: Iterable[float]
    trades: Iterable[Mapping[str, Any]] = ()
    stats: PortfolioStats | None = None

    def serialisable(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "equity_curve": list(self.equity_curve),
            "returns": list(self.returns),
            "trades": [dict(trade) for trade in self.trades],
        }
        if self.stats is not None:
            payload["metrics"] = {
                "sharpe": self.stats.sharpe,
                "sortino": self.stats.sortino,
                "max_drawdown": self.stats.max_drawdown,
                "cvar_95": self.stats.cvar_95,
            }
        return payload


RunnerFn = Callable[[], Mapping[str, Any]]


@dataclass(kw_only=True)
class BacktestHarness:
    """Coordinate backtest execution and result persistence."""

    output_dir: Path
    filename_prefix: str = "backtest"
    metadata: BacktestContext | None = None
    time_provider: Callable[[], float] = time.time

    def run(self, runner: RunnerFn) -> Path:
        """Execute *runner* and persist its results."""

        payload = runner()
        if "returns" not in payload:
            raise KeyError("runner payload must include a 'returns' sequence")
        if "equity_curve" not in payload:
            raise KeyError("runner payload must include an 'equity_curve' sequence")

        returns = list(payload["returns"])
        equity_curve = list(payload["equity_curve"])
        trades = payload.get("trades") or []

        stats = compute_portfolio_stats(returns)
        result = BacktestResult(
            equity_curve=equity_curve, returns=returns, trades=trades, stats=stats
        )

        meta: MutableMapping[str, Any] = {
            "git_sha": _detect_git_sha(),
            "captured_at": self.time_provider(),
        }
        if self.metadata is not None:
            meta.update(self.metadata.as_dict())

        extra_payload: MutableMapping[str, Any] = {
            "metadata": meta,
            "result": result.serialisable(),
        }
        for key, value in payload.items():
            if key in {"returns", "equity_curve", "trades"}:
                continue
            extra_payload[key] = value

        path = save_backtest_results(extra_payload, self.output_dir, prefix=self.filename_prefix)
        return path


__all__ = ["BacktestHarness", "BacktestContext", "BacktestResult"]
