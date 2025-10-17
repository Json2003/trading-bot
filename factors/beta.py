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
        Rolling beta aligned with the shared index of ``asset_returns`` and
        ``market_returns``.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")

    if min_periods is None:
        min_periods = window

    aligned = pd.concat([asset_returns, market_returns], axis=1, join="inner")
    aligned.columns = ["asset", "market"]
    aligned = aligned.dropna()

    if aligned.empty:
        return pd.Series(dtype=float)

    cov = aligned["asset"].rolling(window, min_periods=min_periods).cov(aligned["market"])
    var = aligned["market"].rolling(window, min_periods=min_periods).var()

    beta = cov / var
    beta.name = "beta"
    return beta
