"""Cross-sectional momentum factor (Comp-M) implementation."""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd


def compute_comp_m_factor(
    prices: pd.DataFrame,
    lookback: int,
    lag: int = 1,
    neutralize: bool = True,
) -> pd.DataFrame:
    """Return the Comp-M cross-sectional momentum z-scores.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price table indexed by timestamp with tickers along the columns.
    lookback : int
        Number of periods used for the momentum window.
    lag : int, optional
        How many most recent bars to skip to avoid look-ahead bias. Defaults to 1.
    neutralize : bool, optional
        If ``True``, demean scores each period to keep the factor dollar-neutral.

    Returns
    -------
    pd.DataFrame
        Cross-sectional z-scores with mean zero (if ``neutralize`` is ``True``).
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if lag < 0:
        raise ValueError("lag must be non-negative")

    columns = list(prices.columns)
    index = list(prices.index)
    if not columns or not index:
        return pd.DataFrame({col: [] for col in columns}, index=index)

    price_history: Dict[str, List[float]] = {}
    for col in columns:
        series = prices[col]
        history: List[float] = []
        for value in series:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                history.append(float("nan"))
            else:
                history.append(float(value))
        price_history[col] = history

    momentum_values: Dict[str, List[float]] = {col: [float("nan")] * len(index) for col in columns}

    for t in range(len(index)):
        src = t - lag
        base = src - lookback
        if src < 0 or base < 0:
            continue
        for col in columns:
            series = price_history[col]
            recent = series[src]
            prior = series[base]
            if not _is_positive(recent) or not _is_positive(prior):
                continue
            momentum_values[col][t] = math.log(recent / prior)

    zscore_values: Dict[str, List[float]] = {col: [float("nan")] * len(index) for col in columns}

    for t in range(len(index)):
        raw_vals: List[float] = []
        valid_cols: List[str] = []
        for col in columns:
            value = momentum_values[col][t]
            if math.isnan(value):
                continue
            raw_vals.append(value)
            valid_cols.append(col)
        if not raw_vals:
            continue

        mean_val = sum(raw_vals) / len(raw_vals)
        if neutralize:
            centered = [val - mean_val for val in raw_vals]
        else:
            centered = raw_vals[:]

        variance = sum(val * val for val in centered) / len(centered)
        if not math.isfinite(variance) or variance == 0.0:
            denom = None
        else:
            denom = math.sqrt(variance)

        for idx_col, raw_val, centered_val in zip(valid_cols, raw_vals, centered):
            if denom is None or denom == 0.0:
                zscore = 0.0
            else:
                numerator = centered_val if neutralize else raw_val
                zscore = numerator / denom
            zscore_values[idx_col][t] = zscore

    valid_rows = []
    for t in range(len(index)):
        keep = False
        for col in columns:
            val = zscore_values[col][t]
            if not math.isnan(val):
                keep = True
                break
        valid_rows.append(keep)

    filtered_index = [idx for idx, keep in zip(index, valid_rows) if keep]
    filtered_data = {
        col: [value for value, keep in zip(values, valid_rows) if keep]
        for col, values in zscore_values.items()
    }

    return pd.DataFrame(filtered_data, index=filtered_index)


def _is_positive(value: float) -> bool:
    return value is not None and value > 0 and math.isfinite(value)
