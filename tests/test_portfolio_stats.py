import math

import pytest

from tradingbot_core.metrics import PortfolioStats, compute_portfolio_stats


def _downside_std(values):
    downside = [v for v in values if v < 0]
    if not downside:
        return 0.0
    mean = sum(downside) / len(downside)
    variance = sum((v - mean) ** 2 for v in downside) / len(downside)
    return math.sqrt(variance)


def _max_drawdown(values):
    cumulative = []
    total = 1.0
    for r in values:
        total *= 1 + r
        cumulative.append(total)
    peak = cumulative[0]
    max_dd = 0.0
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = 1 - value / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def _cvar(values):
    sorted_returns = sorted(values)
    position = 0.05 * (len(sorted_returns) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if upper == lower:
        percentile = sorted_returns[int(position)]
    else:
        weight = position - lower
        percentile = sorted_returns[lower] * (1 - weight) + sorted_returns[upper] * weight
    tail = [v for v in sorted_returns if v <= percentile] or [percentile]
    return -sum(tail) / len(tail)


def test_compute_portfolio_stats_basic():
    returns = [0.01, 0.02, -0.015, 0.005, -0.03]

    stats = compute_portfolio_stats(returns, risk_free=0.0)

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    downside_std = _downside_std(returns)

    assert isinstance(stats, PortfolioStats)
    assert math.isclose(stats.sharpe, mean / max(std_dev, 1e-12), rel_tol=1e-6)
    assert math.isclose(stats.sortino, mean / max(downside_std, 1e-12), rel_tol=1e-6)
    assert math.isclose(stats.max_drawdown, _max_drawdown(returns), rel_tol=1e-6)
    assert math.isclose(stats.cvar_95, _cvar(returns), rel_tol=1e-6)


def test_compute_portfolio_stats_requires_data():
    with pytest.raises(ValueError):
        compute_portfolio_stats([])
