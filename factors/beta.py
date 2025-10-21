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
        Rolling beta aligned to ``asset_returns`` with missing values filled
        with ``NaN`` where an estimate cannot be computed.
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
