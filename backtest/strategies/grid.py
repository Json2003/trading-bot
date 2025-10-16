"""Simple grid trading strategy.

This module provides a :func:`generate_signals` function that implements a very
lightweight grid strategy: it builds evenly spaced price levels around the
average close and emits a long (+1) signal when price trades in the lower half
of the grid, a short (-1) signal when it trades in the upper half, and remains
flat (0) near the centre.

The implementation deliberately avoids heavy dependencies so it works with the
light-weight NumPy/Pandas shims that ship with the project.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _resolve_price_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first matching price column from *candidates*.

    The helper accepts several capitalisation variants so the strategy works
    with data sourced from different providers that may use ``close`` or
    ``Close`` column names.  A :class:`KeyError` is raised if none of the
    columns are present.
    """

    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError("No price column found. Tried: " + ", ".join(candidates))


def generate_signals(
    df: pd.DataFrame,
    levels: int = 10,
    range_pct: float = 0.05,
    price_col: str = "close",
) -> pd.DataFrame:
    """Generate grid-trading signals for the provided price series.

    Parameters
    ----------
    df:
        DataFrame containing at least a close-price column.
    levels:
        Number of evenly spaced grid levels to evaluate.  Must be two or more.
    range_pct:
        Fractional range around the mid price to cover with the grid.  For
        example, ``0.05`` builds a grid spanning ±5% around the mid price.
    price_col:
        Name of the close-price column.  The function is case tolerant and will
        also try ``price_col.lower()``, ``price_col.upper()`` and
        ``price_col.capitalize()``.

    Returns
    -------
    pandas.DataFrame
        DataFrame aligned to ``df``'s index containing a ``signals`` column
        with values ``{1, 0, -1}`` for long, flat, and short indications.
    """

    level_count = int(levels)
    if level_count < 2:
        raise ValueError("levels must be at least 2 to build a grid")

    price_candidates = (
        price_col,
        price_col.lower(),
        price_col.upper(),
        price_col.capitalize(),
    )
    resolved_col = _resolve_price_column(df, price_candidates)
    series = df[resolved_col]
    price_list = []
    for value in series:
        try:
            price_list.append(float(value))
        except (TypeError, ValueError):
            price_list.append(float("nan"))

    valid_values = [val for val in price_list if not math.isnan(val)]
    if not valid_values:
        return pd.DataFrame(
            {"signals": np.zeros(len(price_list), dtype=int)}, index=getattr(df, "index", None)
        )

    mid_price = float(sum(valid_values) / len(valid_values))
    if not math.isfinite(mid_price):
        # Degenerate input (all NaN). Return a flat signal series.
        return pd.DataFrame(
            {"signals": np.zeros(len(price_list), dtype=int)}, index=getattr(df, "index", None)
        )

    span = float(range_pct)
    if span < 0:
        raise ValueError("range_pct must be non-negative")

    lower = mid_price * (1 - span)
    upper = mid_price * (1 + span)

    grid_levels = np.linspace(lower, upper, level_count)

    signals_list = []
    for price in price_list:
        if math.isnan(price):
            signals_list.append(0)
            continue
        below = sum(price > level for level in grid_levels)
        above = sum(price < level for level in grid_levels)
        diff = above - below
        if diff > 0:
            signals_list.append(1)
        elif diff < 0:
            signals_list.append(-1)
        else:
            signals_list.append(0)

    signals = np.asarray(signals_list, dtype=int)

    return pd.DataFrame({"signals": signals}, index=getattr(df, "index", None))


__all__ = ["generate_signals"]
