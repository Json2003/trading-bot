"""Utilities for computing rolling beta exposures."""

from __future__ import annotations

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
        Rolling beta aligned to ``asset_returns`` with missing values filled
        with ``NaN`` where an estimate cannot be computed.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")

    if min_periods is None:
        min_periods = window

    market_index = set(market_returns.index)

    def _valid(value: float | int | None) -> bool:
        return value is not None and value == value

    history_asset: list[float] = []
    history_market: list[float] = []
    result: list[float] = []

    for label in asset_returns.index:
        if label not in market_index:
            result.append(float("nan"))
            continue

        asset_value = asset_returns[label]
        market_value = market_returns[label]
        if not _valid(asset_value) or not _valid(market_value):
            result.append(float("nan"))
            continue

        history_asset.append(float(asset_value))
        history_market.append(float(market_value))
        if len(history_asset) > window:
            del history_asset[0]
            del history_market[0]

        if len(history_asset) < min_periods:
            result.append(float("nan"))
            continue

        n = len(history_asset)
        mean_asset = sum(history_asset) / n
        mean_market = sum(history_market) / n
        cov = sum(
            (a - mean_asset) * (m - mean_market)
            for a, m in zip(history_asset, history_market)
        ) / n
        var = sum((m - mean_market) ** 2 for m in history_market) / n

        if var <= 0:
            result.append(float("nan"))
        else:
            result.append(cov / var)

    series = pd.Series(result, index=asset_returns.index)
    series.name = "beta"
    return series
