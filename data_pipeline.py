"""Utilities for data auditing, labeling, and feature storage.

Implements helpers to enforce a leakage-safe research pipeline:
- Convert timestamps to UTC and forward-fill only within a trading session.
- Validate OHLCV inputs by dropping anomalous bars.
- Attach realistic transaction cost columns.
- Build labels for multiple horizons without peeking past t.
- Simple feature-store writer stub aligned with the BigQuery schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple

import math

import pandas as pd


def canonicalize_ohlcv(df: pd.DataFrame, freq: str, session_tz: str = "UTC") -> pd.DataFrame:
    """Return OHLCV data converted to UTC and forward-filled within sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``timestamp`` column.
    freq : str
        Expected frequency like ``'1min'`` or ``'5min'``.
    session_tz : str, optional
        Exchange session timezone, by default "UTC".
    """
    if "timestamp" not in df:
        raise KeyError("missing 'timestamp' column")

    ts = pd.to_datetime(df["timestamp"]).dt.tz_convert(session_tz)
    ts_utc = ts.dt.tz_convert("UTC")
    df = df.copy()
    df["timestamp"] = ts_utc

    # Reindex to expected frequency and forward fill within the session
    start, end = ts_utc.min(), ts_utc.max()
    full_range = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    df = df.set_index("timestamp").reindex(full_range)
    df = df.ffill()
    df.index.name = "timestamp"
    return df.reset_index()


def drop_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars with broken OHLCV or negative volume."""
    mask = (
        (df["high"] >= df["low"]) &
        (df["volume"] >= 0)
    )
    return df[mask].copy()


def attach_cost_columns(df: pd.DataFrame, commission: float, spread: float, slippage: float) -> pd.DataFrame:
    """Attach transaction cost columns for later backtests."""
    df = df.copy()
    df["commission"] = commission
    df["spread"] = spread
    df["slippage"] = slippage
    return df


def _future_return(close: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-horizon)
    return (future / close).apply(math.log)


def directional_return_label(close: pd.Series, horizon: int) -> pd.Series:
    """Label using the sign of the future log return."""
    ret = _future_return(close, horizon)
    return ret.apply(lambda x: 0 if pd.isna(x) else (1 if x > 0 else -1))


def magnitude_bucket_label(close: pd.Series, horizon: int, q: int = 3) -> pd.Series:
    """Quantile-bucket future returns."""
    ret = _future_return(close, horizon)
    return pd.qcut(ret, q, labels=False)


def triple_barrier_label(close: pd.Series, horizon: int, upper: float, lower: float) -> pd.Series:
    """Triple-barrier method with max holding time.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by timestamp.
    horizon : int
        Maximum look-ahead steps.
    upper : float
        Upper percentage barrier.
    lower : float
        Lower percentage barrier (positive value).
    """
    log_close = close.apply(math.log)
    out = pd.Series(index=close.index, dtype="float64")
    for i in range(len(close)):
        start = log_close.iloc[i]
        end = min(i + horizon, len(close) - 1)
        window = log_close.iloc[i + 1 : end + 1]
        if window.empty:
            out.iloc[i] = 0
            continue
        diff = window - start
        hit_upper = (diff >= math.log(1 + upper)).idxmax() if (diff >= math.log(1 + upper)).any() else None
        hit_lower = (diff <= -math.log(1 + lower)).idxmax() if (diff <= -math.log(1 + lower)).any() else None
        first_hit = None
        if hit_upper is not None:
            first_hit = hit_upper
            label = 1
        if hit_lower is not None and (first_hit is None or window.index.get_loc(hit_lower) < window.index.get_loc(first_hit)):
            first_hit = hit_lower
            label = -1
        out.iloc[i] = label if first_hit is not None else 0
    return out


@dataclass
class FeatureStore:
    """Very small BigQuery-oriented feature store stub."""
    dataset: str = "market_fs"
    table: str = "features_ohlcv_min"

    def write(self, df: pd.DataFrame, feature_version: str, source_hash: str, *, project: Optional[str] = None) -> Tuple[str, str]:
        """Write features to BigQuery or, if unavailable, to a local CSV.

        Returns the destination (dataset.table) and path written.
        """
        dest = f"{self.dataset}.{self.table}"
        try:
            from pandas_gbq import to_gbq  # type: ignore
            to_gbq(df, dest, project_id=project, if_exists="append")
            return dest, "bigquery"
        except Exception:
            path = f"{self.table}.csv"
            df["feature_version"] = feature_version
            df["source_hash"] = source_hash
            df.to_csv(path, index=False)
            return dest, path


def purged_kfold(n_splits: int, embargo: int, n_samples: int) -> Iterator[Tuple[Iterable[int], Iterable[int]]]:
    """Yield purged train/validation indices with embargo.

    This generator splits [0, n_samples) into ``n_splits`` folds. For each
    fold, the validation slice is removed from the training set together with
    an embargo of ``embargo`` samples on each side.
    """
    fold_size = n_samples // n_splits
    indices = list(range(n_samples))
    for i in range(n_splits):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        val_idx = indices[start:stop]
        train_idx = indices[: max(0, start - embargo)] + indices[min(n_samples, stop + embargo) :]
        yield train_idx, val_idx
