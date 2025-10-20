"""Cross-sectional momentum factor (Comp-M) implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_comp_m_factor(
    prices: pd.DataFrame,
    lookback: int,
    lag: int = 1,
    neutralize: bool = True,
) -> pd.DataFrame:
    """Return cross-sectional momentum z-scores for each timestamp.

    The Comp-M factor compares the trailing ``lookback`` period return for each
    asset against the rest of the universe.  The series is shifted by ``lag`` to
    avoid look-ahead bias and optionally demeaned each period so that the factor
    tilts are dollar-neutral on average.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if len(prices) == 0:
        return prices.copy()

    columns = list(getattr(prices, "columns", []))
    index = list(getattr(prices, "index", []))
    data = {col: list(prices[col]) for col in columns}

    momentum_rows: list[dict[str, float | None]] = []
    for row_idx, label in enumerate(index):
        row: dict[str, float | None] = {}
        for col in columns:
            series = data[col]
            if row_idx - lookback < 0:
                row[col] = None
                continue
            current = series[row_idx]
            previous = series[row_idx - lookback]
            if previous in (0, None) or previous != previous or current != current:
                row[col] = None
                continue
            row[col] = (float(current) - float(previous)) / float(previous)
        momentum_rows.append(row)

    if lag:
        shifted: list[dict[str, float | None]] = [
            {} for _ in range(len(momentum_rows))
        ]
        for idx_row in range(len(momentum_rows)):
            src = idx_row - lag
            shifted[idx_row] = momentum_rows[src] if src >= 0 else {col: None for col in columns}
        momentum_rows = shifted

    zscore_rows: list[dict[str, float]] = []
    for row in momentum_rows:
        values = [value for value in row.values() if value is not None]
        if not values:
            zscore_rows.append({col: np.nan for col in columns})
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = float(np.sqrt(variance)) if variance > 0 else 0.0

        output_row: dict[str, float] = {}
        for col, value in row.items():
            if value is None or std_value == 0.0:
                output_row[col] = np.nan
                continue
            adjusted = value - mean_value if neutralize else value
            output_row[col] = adjusted / std_value
        zscore_rows.append(output_row)

    result_data = {col: [] for col in columns}
    for row in zscore_rows:
        for col in columns:
            result_data[col].append(row.get(col, np.nan))

    frame = pd.DataFrame(result_data, index=index)

    if hasattr(frame, "dropna"):
        return frame.dropna(how="all")
    return frame
