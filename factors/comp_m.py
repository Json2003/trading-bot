"""Cross-sectional momentum factor (Comp-M) implementation."""

from __future__ import annotations

import math
<<<<<<< HEAD

import pandas as pd


=======
import numpy as np
import pandas as pd


def _zscore_frame(values: pd.DataFrame, neutralize: bool) -> pd.DataFrame:
    """Return row-wise z-scores with optional cross-sectional demeaning."""

    if getattr(values, "empty", False):
        return values.copy()

    demeaned = values
    if neutralize and hasattr(values, "mean"):
        demean = values.mean(axis=1, skipna=True)
        demeaned = values.sub(demean, axis=0)

    if hasattr(values, "std"):
        std = values.std(axis=1, skipna=True, ddof=0)
        std_replaced = std.replace(0.0, np.nan)
        return demeaned.div(std_replaced, axis=0)

    # Fallback for minimal DataFrame implementations without vectorised stats.
    columns = list(getattr(values, "columns", []))
    index = list(getattr(values, "index", range(len(values))))
    rows: list[dict[str, float]] = []
    for row_idx, _ in enumerate(index):
        row = {col: values[col][row_idx] for col in columns}
        valid = [v for v in row.values() if v == v]
        if not valid:
            rows.append({col: np.nan for col in columns})
            continue
        mean_value = sum(valid) / len(valid)
        variance = sum((val - mean_value) ** 2 for val in valid) / len(valid)
        std_value = float(np.sqrt(variance)) if variance > 0 else float("nan")
        out_row: dict[str, float] = {}
        for col, value in row.items():
            if value != value or std_value != std_value:
                out_row[col] = np.nan
                continue
            adjusted = value - mean_value if neutralize else value
            if std_value == 0 or std_value != std_value:
                out_row[col] = np.nan
            else:
                out_row[col] = adjusted / std_value
        rows.append(out_row)
    return pd.DataFrame(rows, index=index)


>>>>>>> origin/main
def compute_comp_m_factor(
    prices: pd.DataFrame,
    lookback: int,
    lag: int = 1,
<<<<<<< HEAD
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
=======
    skip: int = 0,
    neutralize: bool = True,
) -> pd.DataFrame:
    """Return cross-sectional momentum z-scores for each timestamp.

    The Comp-M factor compares trailing log returns across the universe,
    optionally skipping the most recent ``skip`` bars to reduce noise and
    shifting by ``lag`` to avoid look-ahead bias.  Values are standardized on
    each timestamp so that the average tilt remains roughly neutral.
>>>>>>> origin/main
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if lag < 0:
        raise ValueError("lag must be non-negative")
<<<<<<< HEAD

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
=======
    if skip < 0:
        raise ValueError("skip must be non-negative")

    if hasattr(prices, "empty") and prices.empty:
        return prices.copy()

    if hasattr(prices, "diff") and hasattr(prices, "shift"):
        float_prices = prices.astype(float)
        if hasattr(np, "log"):
            log_prices = np.log(float_prices)
        else:  # pragma: no cover - used in environments without full numpy
            log_prices = float_prices.applymap(math.log)
        momentum = log_prices.diff(periods=lookback)

        if skip:
            momentum = momentum.shift(skip)

        if lag:
            momentum = momentum.shift(lag)

        return _zscore_frame(momentum, neutralize=neutralize)

    # Fallback path for the lightweight pandas stub used in tests.  It mirrors the
    # vectorised implementation above but works with plain Python data.
    columns = list(getattr(prices, "columns", []))
    index = list(getattr(prices, "index", range(len(prices))))
    if not columns:
        return pd.DataFrame(index=index)

    data = {col: list(prices[col]) for col in columns}

    momentum_rows: list[dict[str, float]] = []
    for row_idx, _ in enumerate(index):
        src_idx = row_idx - lag
        eval_idx = src_idx - skip if src_idx is not None else None
        row: dict[str, float] = {}
        for col in columns:
            series = data[col]
            if src_idx is None or src_idx < 0 or eval_idx is None or eval_idx < 0:
                row[col] = np.nan
                continue
            if eval_idx >= len(series) or eval_idx - lookback < 0:
                row[col] = np.nan
                continue
            current = series[eval_idx]
            previous = series[eval_idx - lookback]
            if current is None or previous in (None, 0):
                row[col] = np.nan
                continue
            try:
                current_f = float(current)
                previous_f = float(previous)
            except Exception:
                row[col] = np.nan
                continue
            if current_f <= 0 or previous_f <= 0:
                row[col] = np.nan
                continue
            try:
                row[col] = float(math.log(current_f) - math.log(previous_f))
            except Exception:
                row[col] = np.nan
        momentum_rows.append(row)

    z_rows: list[dict[str, float]] = []
    for row in momentum_rows:
        values = [v for v in row.values() if v == v]
        if not values:
            z_rows.append({col: np.nan for col in columns})
            continue
        mean_value = sum(values) / len(values)
        variance = sum((val - mean_value) ** 2 for val in values) / len(values)
        std_value = float(np.sqrt(variance)) if variance > 0 else float("nan")
        out_row: dict[str, float] = {}
        for col, value in row.items():
            if value != value or std_value != std_value:
                out_row[col] = np.nan
                continue
            adjusted = value - mean_value if neutralize else value
            if std_value == 0 or std_value != std_value:
                out_row[col] = np.nan
            else:
                out_row[col] = adjusted / std_value
        z_rows.append(out_row)

    return pd.DataFrame(z_rows, index=index)
>>>>>>> origin/main
