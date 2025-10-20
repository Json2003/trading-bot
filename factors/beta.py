"""Utilities for computing rolling beta exposures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _as_float_series(values: pd.Series) -> pd.Series:
    """Return ``values`` converted to floats while preserving the index."""

    try:
        return values.astype(float)
    except Exception:
        index = list(getattr(values, "index", []))
        data: list[float] = []
        for item in values:
            try:
                data.append(float(item))
            except Exception:
                data.append(math.nan)
        if not index:
            index = list(range(len(data)))
        return pd.Series(data, index=index, dtype=float)


def _manual_beta(
    aligned: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Fallback beta implementation for reduced pandas environments."""

    asset = list(aligned["asset"])
    market = list(aligned["market"])
    beta: list[float] = [math.nan] * len(asset)

    for pos in range(len(aligned)):
        start = max(0, pos - window + 1)
        length = pos - start + 1
        if length < min_periods:
            continue

        asset_slice = asset[start : pos + 1]
        market_slice = market[start : pos + 1]

        asset_mean = float(sum(asset_slice) / length)
        market_mean = float(sum(market_slice) / length)

        cov = sum((a - asset_mean) * (m - market_mean) for a, m in zip(asset_slice, market_slice))
        cov /= length
        var = sum((m - market_mean) ** 2 for m in market_slice) / length
        if not math.isfinite(var) or abs(var) <= 1e-18:
            continue

        beta[pos] = cov / var

    index = list(getattr(aligned, "index", []))
    if len(index) != len(beta):
        index = list(range(len(beta)))
    return pd.Series(beta, index=index, dtype=float)


def _align_series(asset_returns: pd.Series, market_returns: pd.Series) -> pd.DataFrame:
    try:
        combined = pd.DataFrame({"asset": asset_returns, "market": market_returns})
        combined = combined.dropna()
        combined = combined.apply(pd.to_numeric, errors="coerce")
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
        try:
            if len(combined) == 0:
                raise ValueError
        except TypeError:
            raise ValueError
        return combined
    except Exception:
        pass

    asset_index = list(getattr(asset_returns, "index", []))
    market_index = list(getattr(market_returns, "index", []))
    market_values = {}
    for idx in market_index:
        try:
            value = float(market_returns[idx])
        except Exception:
            continue
        if math.isfinite(value):
            market_values[idx] = value

    aligned_idx: list = []
    asset_vals: list[float] = []
    market_vals: list[float] = []

    for idx in asset_index:
        if idx not in market_values:
            continue
        try:
            asset_val = float(asset_returns[idx])
        except Exception:
            continue
        if not math.isfinite(asset_val):
            continue
        aligned_idx.append(idx)
        asset_vals.append(asset_val)
        market_vals.append(market_values[idx])

    if not aligned_idx:
        return pd.DataFrame({"asset": [], "market": []})

    return pd.DataFrame({"asset": asset_vals, "market": market_vals}, index=aligned_idx)


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
        Minimum observations required for a beta estimate. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Rolling beta aligned with the shared index of ``asset_returns`` and
        ``market_returns``. The returned series is reindexed to ``asset_returns``
        so callers retain their original alignment.
    """

    if window <= 1:
        raise ValueError("window must be greater than 1")

    min_periods = int(min_periods or window)
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")

    aligned = _align_series(asset_returns, market_returns)
    try:
        if len(aligned) == 0:
            return pd.Series(index=asset_returns.index, dtype=float)
    except TypeError:
        return pd.Series(index=asset_returns.index, dtype=float)

    try:
        asset = _as_float_series(aligned["asset"])
        market = _as_float_series(aligned["market"])

        rolling_cov = asset.rolling(window, min_periods=min_periods).cov(market)
        rolling_var = market.rolling(window, min_periods=min_periods).var()
        rolling_var = rolling_var.replace(0.0, np.nan)
        beta = rolling_cov / rolling_var
    except Exception:
        beta = _manual_beta(aligned, window=window, min_periods=min_periods)

    beta = beta.replace([-np.inf, np.inf], np.nan)
    if hasattr(beta, "reindex"):
        try:
            return beta.reindex(asset_returns.index)
        except Exception:
            pass

    asset_index = list(getattr(asset_returns, "index", []))
    beta_index = list(getattr(beta, "index", []))
    beta_values = list(beta)
    lookup = {idx: val for idx, val in zip(beta_index, beta_values)}
    if not asset_index:
        asset_index = list(range(len(asset_returns)))
    aligned_values = [lookup.get(idx, math.nan) for idx in asset_index]
    return pd.Series(aligned_values, index=asset_index, dtype=float)
