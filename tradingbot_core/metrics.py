"""Common performance metrics used in reports and dashboards."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class Drawdown:
    peak: float
    trough: float
    recovery: float
    depth: float


def _validate_series(series: Sequence[float]) -> None:
    if not series:
        raise ValueError("Expected a non-empty series")


def calculate_cumulative_returns(returns: Iterable[float]) -> list[float]:
    cumulative = []
    total = 1.0
    for value in returns:
        total *= 1.0 + value
        cumulative.append(total - 1.0)
    return cumulative


def calculate_max_drawdown(equity_curve: Sequence[float]) -> Drawdown:
    _validate_series(equity_curve)
    peak = equity_curve[0]
    trough = equity_curve[0]
    max_depth = 0.0
    recovery = equity_curve[0]

    for value in equity_curve[1:]:
        if value > peak:
            peak = value
            trough = value
        elif value < trough:
            trough = value
            depth = (trough - peak) / peak if peak else 0.0
            if depth < max_depth:
                max_depth = depth
                recovery = value
    return Drawdown(peak=peak, trough=trough, recovery=recovery, depth=max_depth)


def calculate_sharpe_ratio(returns: Sequence[float], *, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    _validate_series(returns)
    mean_return = sum(returns) / len(returns)
    excess_returns = [r - risk_free_rate / periods_per_year for r in returns]
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        raise ValueError("Standard deviation is zero; Sharpe ratio undefined")
    return (mean_return - risk_free_rate / periods_per_year) / std_dev * math.sqrt(periods_per_year)


def calculate_sortino_ratio(returns: Sequence[float], *, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    _validate_series(returns)
    downside = [min(0.0, r - risk_free_rate / periods_per_year) for r in returns]
    downside_squared = [r ** 2 for r in downside]
    downside_deviation = math.sqrt(sum(downside_squared) / len(returns))
    if downside_deviation == 0:
        raise ValueError("Downside deviation is zero; Sortino ratio undefined")
    mean_return = sum(returns) / len(returns)
    return (mean_return - risk_free_rate / periods_per_year) / downside_deviation * math.sqrt(periods_per_year)


__all__ = [
    "Drawdown",
    "calculate_cumulative_returns",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
]
