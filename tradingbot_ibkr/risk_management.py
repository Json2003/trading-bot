"""Risk management helpers for the trading engine."""
from __future__ import annotations
import numpy as np
import pandas as pd


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Compute Conditional Value at Risk (CVaR).

    Args:
        returns: 1D array of returns.
        alpha: Confidence level.
    """
    if len(returns) == 0:
        return 0.0
    var = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())


def apply_drawdown_limit(equity_curve: pd.Series, max_drawdown: float) -> bool:
    """Return True if max drawdown is breached."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min() <= -abs(max_drawdown)


def volatility_filter(prices: pd.Series, threshold: float) -> bool:
    """Return True if volatility is below threshold."""
    if len(prices) < 2:
        return True
    vol = prices.pct_change().std()
    return vol < threshold
