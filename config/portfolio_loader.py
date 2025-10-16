"""Lightweight helpers for reading the portfolio YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "PortfolioConfig",
    "StrategyConfig",
    "load_portfolio_config",
    "load_strategy_params",
]


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration block describing a single strategy allocation."""

    name: str
    capital: float
    max_position_notional: float | None = None
    max_drawdown: float | None = None


@dataclass(frozen=True)
class PortfolioConfig:
    """Top-level configuration for the multi-strategy portfolio."""

    base_currency: str
    total_capital: float
    portfolio_limits: Mapping[str, float]
    strategies: tuple[StrategyConfig, ...]

    def allocation_for(self, name: str) -> StrategyConfig:
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        raise KeyError(f"Unknown strategy {name!r}")


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    """Load the structured portfolio configuration from ``path``."""

    raw_path = Path(path)
    data = yaml.safe_load(raw_path.read_text()) or {}
    try:
        base_currency = str(data.get("base_currency", "USD"))
        total_capital = float(data["total_capital"])
    except KeyError as exc:  # pragma: no cover - configuration error
        raise KeyError(f"Missing required portfolio setting: {exc.args[0]}") from exc

    portfolio_limits = {
        str(key): float(value)
        for key, value in (data.get("portfolio_limits") or {}).items()
    }

    strategy_blocks = data.get("strategies") or {}
    strategies = []
    for name, payload in strategy_blocks.items():
        capital = float(payload["capital"])
        max_position_notional = (
            float(payload["max_position_notional"])
            if payload.get("max_position_notional") is not None
            else None
        )
        max_drawdown = (
            float(payload["max_drawdown"])
            if payload.get("max_drawdown") is not None
            else None
        )
        strategies.append(
            StrategyConfig(
                name=str(name),
                capital=capital,
                max_position_notional=max_position_notional,
                max_drawdown=max_drawdown,
            )
        )

    return PortfolioConfig(
        base_currency=base_currency,
        total_capital=total_capital,
        portfolio_limits=portfolio_limits,
        strategies=tuple(strategies),
    )


def load_strategy_params(name: str, strategy_dir: str | Path) -> dict[str, Any]:
    """Return the raw YAML configuration for ``name`` from ``strategy_dir``."""

    path = Path(strategy_dir) / f"{name}.yaml"
    if not path.exists():  # pragma: no cover - configuration error
        raise FileNotFoundError(f"Missing strategy configuration file: {path}")
    return yaml.safe_load(path.read_text()) or {}
