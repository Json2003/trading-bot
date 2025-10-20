"""Cross-sectional momentum factor (Comp-M) implementation."""

from __future__ import annotations

import math

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

    columns = list(getattr(prices, "columns", []))
    index = list(getattr(prices, "index", []))
    try:
        row_count = len(prices)
    except TypeError:
        row_count = 0

    if row_count == 0 or not columns:
        return pd.DataFrame({col: [] for col in columns}, index=index)

    price_history: dict[str, list[float]] = {}
    for col in columns:
        series = prices[col]
        history: list[float] = []
        for value in series:
            try:
                history.append(float(value))
            except Exception:
                history.append(float("nan"))
        price_history[col] = history

    momentum: dict[str, list[float]] = {col: [float("nan")] * row_count for col in columns}

    for t in range(row_count):
        src = t - lag
        base = src - lookback
        if src < 0 or base < 0:
            continue
        for col in columns:
            recent = price_history[col][src]
            prior = price_history[col][base]
            if not _is_positive(recent) or not _is_positive(prior):
                continue
            momentum[col][t] = math.log(recent / prior)

    zscores: dict[str, list[float]] = {col: [float("nan")] * row_count for col in columns}

    for t in range(row_count):
        values: list[float] = []
        valid_cols: list[str] = []
        for col in columns:
            value = momentum[col][t]
            if math.isnan(value):
                continue
            values.append(value)
            valid_cols.append(col)
        if not values:
            continue

        mean_val = sum(values) / len(values)
        if neutralize:
            centered = [val - mean_val for val in values]
        else:
            centered = values[:]

        variance = sum(val * val for val in centered) / len(centered)
        if not math.isfinite(variance) or variance <= 1e-12:
            for col in valid_cols:
                zscores[col][t] = 0.0
            continue

        denom = math.sqrt(variance)
        for col, raw_val, centered_val in zip(valid_cols, values, centered):
            numerator = centered_val if neutralize else raw_val
            zscores[col][t] = numerator / denom

    data = {col: zscores[col] for col in columns}
    return pd.DataFrame(data, index=index)


def _is_positive(value: float) -> bool:
    return value is not None and value > 0 and math.isfinite(value)
