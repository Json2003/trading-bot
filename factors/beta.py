"""Utilities for computing rolling beta exposures."""

from __future__ import annotations

import math
import pandas as pd


def compute_rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Return the rolling beta of an asset relative to the market.

    Parameters
    ----------
    asset_returns : pd.Series
        Return stream for the asset you want to hedge, indexed by timestamp.
    market_returns : pd.Series
        Return stream for the benchmark market (e.g. BTC/USDT).
    window : int
        Number of observations to use for the rolling regression.
    min_periods : int, optional
        Minimum observations required for a beta estimate.  Defaults to the
        rolling window size.

    Returns
    -------
    pd.Series
        Rolling beta aligned with the shared index of ``asset_returns`` and
        ``market_returns``.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")

    if min_periods is None:
        min_periods = window

    records: list[tuple] = []
    asset_index = list(asset_returns.index)
    market_data = {idx: market_returns[idx] for idx in market_returns.index}
    for idx in asset_index:
        if idx not in market_data:
            continue
        asset_val = asset_returns[idx]
        market_val = market_data[idx]
        if pd.isna(asset_val) or pd.isna(market_val):
            continue
        records.append((idx, float(asset_val), float(market_val)))

    if not records:
        return pd.Series(dtype=float)

    aligned = pd.DataFrame(
        {
            "asset": [row[1] for row in records],
            "market": [row[2] for row in records],
        },
        index=[row[0] for row in records],
    )

    asset_values = list(aligned["asset"])
    market_values = list(aligned["market"])
    beta_values: list[float] = [float("nan")] * len(aligned.index)

    for i in range(len(aligned.index)):
        start = max(0, i - window + 1)
        length = i - start + 1
        if length < (min_periods or window):
            continue

        asset_window = [float(x) for x in asset_values[start : i + 1]]
        market_window = [float(x) for x in market_values[start : i + 1]]

        asset_mean = sum(asset_window) / len(asset_window)
        market_mean = sum(market_window) / len(market_window)

        cov = sum((a - asset_mean) * (m - market_mean) for a, m in zip(asset_window, market_window))
        cov /= len(asset_window)
        var = sum((m - market_mean) ** 2 for m in market_window) / len(market_window)
        if not math.isfinite(var) or var == 0.0:
            beta_values[i] = float("nan")
            continue
        beta_values[i] = cov / var

    result_index = list(asset_returns.index)
    result_values = [float("nan")] * len(result_index)
    index_to_pos = {idx: pos for pos, idx in enumerate(result_index)}
    for idx, value in zip(aligned.index, beta_values):
        pos = index_to_pos.get(idx)
        if pos is not None:
            result_values[pos] = value
    return pd.Series(result_values, index=result_index, dtype=float)
