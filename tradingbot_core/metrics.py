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


@dataclass(frozen=True)
class PortfolioStats:
    """Aggregate portfolio statistics computed from a return series."""

    sharpe: float
    sortino: float
    max_drawdown: float
    cvar_95: float


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


def calculate_sharpe_ratio(
    returns: Sequence[float], *, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    _validate_series(returns)
    mean_return = sum(returns) / len(returns)
    excess_returns = [r - risk_free_rate / periods_per_year for r in returns]
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        raise ValueError("Standard deviation is zero; Sharpe ratio undefined")
    return (mean_return - risk_free_rate / periods_per_year) / std_dev * math.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: Sequence[float], *, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    _validate_series(returns)
    downside = [min(0.0, r - risk_free_rate / periods_per_year) for r in returns]
    downside_squared = [r**2 for r in downside]
    downside_deviation = math.sqrt(sum(downside_squared) / len(returns))
    if downside_deviation == 0:
        raise ValueError("Downside deviation is zero; Sortino ratio undefined")
    mean_return = sum(returns) / len(returns)
    return (
        (mean_return - risk_free_rate / periods_per_year)
        / downside_deviation
        * math.sqrt(periods_per_year)
    )


def compute_portfolio_stats(returns: Sequence[float], *, risk_free: float = 0.0) -> PortfolioStats:
    """Compute common portfolio statistics from a series of returns.

    Args:
        returns: Iterable of periodic returns expressed as decimals (e.g. ``0.01`` for 1%).
        risk_free: Periodic risk-free rate used for excess return calculations.

    Returns:
        PortfolioStats dataclass containing Sharpe ratio, Sortino ratio, maximum drawdown,
        and Conditional Value at Risk (CVaR) at the 95% level.

    Raises:
        ValueError: If ``returns`` is empty.
    """

    _validate_series(returns)

    values = [float(r) for r in returns]
    eps = 1e-12

    mean_return = sum(values) / len(values)
    variance = sum((r - mean_return) ** 2 for r in values) / len(values)
    std_dev = math.sqrt(variance)
    std_dev = std_dev if std_dev > eps else eps

    downside = [r for r in values if r < 0.0]
    if downside:
        downside_mean = sum(downside) / len(downside)
        downside_var = sum((r - downside_mean) ** 2 for r in downside) / len(downside)
        downside_dev = math.sqrt(downside_var)
    else:
        downside_dev = 0.0
    downside_dev = downside_dev if downside_dev > eps else eps

    sharpe = (mean_return - risk_free) / std_dev
    sortino = (mean_return - risk_free) / downside_dev

    cumulative = []
    total = 1.0
    for r in values:
        total *= 1.0 + r
        cumulative.append(total)

    max_drawdown = 0.0
    peak = cumulative[0] if cumulative else 0.0
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = 1.0 - value / peak if peak else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    sorted_returns = sorted(values)
    if sorted_returns:
        position = 0.05 * (len(sorted_returns) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if upper == lower:
            percentile = sorted_returns[int(position)]
        else:
            weight = position - lower
            percentile = sorted_returns[lower] * (1 - weight) + sorted_returns[upper] * weight
        tail_losses = [r for r in sorted_returns if r <= percentile]
        if not tail_losses:
            tail_losses = [percentile]
        cvar = -sum(tail_losses) / len(tail_losses)
    else:
        cvar = 0.0

    return PortfolioStats(sharpe=sharpe, sortino=sortino, max_drawdown=max_drawdown, cvar_95=cvar)


__all__ = [
    "Drawdown",
    "PortfolioStats",
    "calculate_cumulative_returns",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "compute_portfolio_stats",
]
