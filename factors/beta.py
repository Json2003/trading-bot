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

    # Prefer the vectorised implementation when a full pandas Series/DataFrame is
    # available.  The repo ships with a very small pandas stub for environments
    # without the dependency, so we guard the fast-path behind attribute checks.
    if hasattr(asset_returns, "rolling") and hasattr(market_returns, "reindex"):
        market_aligned = market_returns.reindex(asset_returns.index)
        df = pd.DataFrame({"asset": asset_returns, "market": market_aligned})

        cov = df["asset"].rolling(window=window, min_periods=min_periods).cov(
            df["market"]
        )
        var = df["market"].rolling(window=window, min_periods=min_periods).var()

        beta = cov / var
        beta.name = "beta"
        return beta

    asset_index = list(getattr(asset_returns, "index", range(len(asset_returns))))
    market_index = list(getattr(market_returns, "index", range(len(market_returns))))
    market_map = {label: value for label, value in zip(market_index, list(market_returns))}

    def _valid(value: float | int | None) -> bool:
        return value is not None and value == value

    history_asset: list[float] = []
    history_market: list[float] = []
    result: list[float] = []

    asset_values = list(asset_returns)

    for idx, label in enumerate(asset_index):
        if label not in market_map:
            result.append(float("nan"))
            continue

        asset_value = asset_values[idx] if idx < len(asset_values) else None
        market_value = market_map.get(label)
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

        result.append(float("nan") if var <= 0 else cov / var)

    return pd.Series(result, index=asset_index)
