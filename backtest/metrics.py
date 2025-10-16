"""Common backtest metrics utilities.

Includes max_drawdown, sharpe_ratio, and profit_factor.
Safe to import in this repo where files named pandas.py/requests.py
may exist, by avoiding importing pandas at module import time.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Iterable, TYPE_CHECKING, Dict

if TYPE_CHECKING:  # type hints only
    import pandas as pd  # type: ignore


def max_drawdown(equity):
    """Compute maximum drawdown from an equity curve Series/array.

    equity: pandas Series-like numeric sequence.
    Returns float in [-1, 0].
    """
    values = []
    if hasattr(equity, "_values"):
        values = [float(v) for v in equity._values if v is not None]
    elif hasattr(equity, "values"):
        try:
            values = [float(v) for v in list(equity.values)]
        except Exception:
            values = [float(v) for v in equity]
    else:
        values = [float(v) for v in equity]
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            drawdown = (v / peak) - 1.0
            if drawdown < worst:
                worst = drawdown
    return float(worst)


def _to_float_list(values):
    if hasattr(values, "_values"):
        return [float(v) for v in values._values if v is not None]
    try:
        return [float(v) for v in values]
    except TypeError:
        return [float(values)]


def sharpe_ratio(returns, periods_per_year: int = 365) -> float:
    """Annualized Sharpe ratio from periodic returns Series.

    Uses population std (ddof=0) to match many backtesting libs.
    """
    vals = _to_float_list(returns)
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(variance)
    if std == 0.0 or math.isclose(std, 0.0):
        return 0.0
    return float((mean * periods_per_year) / (std * math.sqrt(periods_per_year)))


def profit_factor(trade_pnls: Iterable[float]) -> float:
    """Sum of wins divided by absolute sum of losses.

    Infinite if no losses and at least one gain; 0 if no gains and no losses.
    """
    gains = 0.0
    losses = 0.0
    for x in trade_pnls:
        if x > 0:
            gains += x
        elif x < 0:
            losses -= x  # accumulate absolute value
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return float(gains / losses)


def sortino_ratio(returns, periods_per_year: int = 365) -> float:
    """Annualized Sortino ratio from periodic returns sequence/Series."""
    vals = _to_float_list(returns)
    if not vals:
        return 0.0
    downside = [x for x in vals if x < 0]
    if not downside:
        return 0.0
    mean = sum(vals) / len(vals)
    down_mean = sum(downside) / len(downside)
    down_variance = sum((x - down_mean) ** 2 for x in downside) / len(downside)
    down_std = math.sqrt(down_variance)
    if down_std == 0.0 or math.isclose(down_std, 0.0):
        return 0.0
    mean_annual = mean * periods_per_year
    down_std_annual = down_std * math.sqrt(periods_per_year)
    return float(mean_annual / down_std_annual)


def summarize(trades, equity_curve, bar_returns, periods_per_year: int = 365) -> Dict:
    """Produce a compact performance summary from trades and equity curve.

    trades: DataFrame-like with column 'pnl' (optional; if missing, PF=0, avg/WR=0)
    equity_curve: DataFrame-like with columns 'timestamp' and 'equity' (equity relative to start, e.g., 1.0 base)
    bar_returns: sequence/Series of per-bar simple returns
    """
    # Extract equity series safely
    try:
        equity_series = equity_curve["equity"]
        ts_series = equity_curve["timestamp"]
        end_equity = (
            float(equity_series[-1])
            if isinstance(equity_series, list)
            else float(getattr(equity_series, "iloc", equity_series)[-1])
        )
        start_ts = (
            str(ts_series[0])
            if isinstance(ts_series, list)
            else str(getattr(ts_series, "iloc", ts_series)[0])
        )
        end_ts = (
            str(ts_series[-1])
            if isinstance(ts_series, list)
            else str(getattr(ts_series, "iloc", ts_series)[-1])
        )
    except Exception:
        arr = np.asarray(equity_curve, dtype=float)
        end_equity = float(arr[-1]) if arr.size else 1.0
        start_ts = ""
        end_ts = ""

    total_ret = end_equity - 1.0
    dd = max_drawdown(
        equity_curve["equity"]
        if isinstance(equity_curve, dict) or hasattr(equity_curve, "__getitem__")
        else equity_curve
    )
    sr = sharpe_ratio(bar_returns, periods_per_year)
    sor = sortino_ratio(bar_returns, periods_per_year)

    # Trades-based metrics
    try:
        pnls = list(trades["pnl"])  # works for list-like or pandas Series
        wr = float(np.mean(np.asarray(pnls) > 0.0)) if len(pnls) else 0.0
        avg = float(np.mean(pnls)) if len(pnls) else 0.0
        pf = profit_factor(pnls) if len(pnls) else 0.0
        num_trades = int(len(pnls))
    except Exception:
        wr = 0.0
        avg = 0.0
        pf = 0.0
        num_trades = 0

    return {
        "total_return": float(total_ret),
        "max_drawdown": float(dd),
        "sharpe": float(sr),
        "sortino": float(sor),
        "profit_factor": float(pf),
        "win_rate": float(wr),
        "avg_trade": float(avg),
        "num_trades": num_trades,
        "start": start_ts,
        "end": end_ts,
    }
