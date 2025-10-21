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
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, Optional, Tuple

import math

import pandas as pd
import numpy as np

try:  # pragma: no cover - fallback branch exercised when zoneinfo missing
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


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

    ts_series = pd.to_datetime(df["timestamp"])

    def _resolve_tz(name: str):
        if ZoneInfo is None:
            return timezone.utc
        try:
            return ZoneInfo(name)
        except Exception:
            if name.upper() == "UTC":
                return timezone.utc
            try:
                return ZoneInfo("UTC")
            except Exception:
                return timezone.utc

    session_tzinfo = _resolve_tz(session_tz)
    utc_values = []
    for value in ts_series._values:
        if not isinstance(value, datetime):
            try:
                value = datetime.fromisoformat(str(value))
            except Exception:
                utc_values.append(value)
                continue
        if value.tzinfo is None:
            localized = value.replace(tzinfo=session_tzinfo)
        else:
            localized = value.astimezone(session_tzinfo)
        utc_values.append(localized.astimezone(timezone.utc))

    df = df.copy()
    df["timestamp"] = utc_values

    valid_times = [ts for ts in utc_values if isinstance(ts, datetime)]
    if len(valid_times) != len(df):
        raise ValueError("timestamp column must contain datetime-like values")
    if not valid_times:
        raise ValueError("timestamp column must contain datetime-like values")

    session_dates = [
        ts.astimezone(session_tzinfo).date() if isinstance(ts, datetime) else None
        for ts in utc_values
    ]
    df["_session_date"] = session_dates

    grouped_records: dict = {}
    for record in df.to_dict("records"):
        key = record.get("_session_date")
        grouped_records.setdefault(key, []).append(record)

    output_records = []
    for key in sorted(grouped_records):
        rows = grouped_records[key]
        if not rows:
            continue
        rows.sort(key=lambda row: row["timestamp"])
        start = rows[0]["timestamp"]
        end = rows[-1]["timestamp"]
        session_df = pd.DataFrame(rows)
        session_df = session_df.set_index("timestamp")
        session_range = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
        reindexed = session_df.reindex(session_range)
        reindexed = reindexed.ffill()
        session_records = reindexed.reset_index().to_dict("records")
        for row in session_records:
            row.pop("_session_date", None)
            if "index" in row:
                row["timestamp"] = row.pop("index")
            output_records.append(row)

    if not output_records:
        raise ValueError("timestamp column must contain datetime-like values")

    return pd.DataFrame(output_records)


def drop_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars with broken OHLCV or negative volume."""
    mask = (df["high"] >= df["low"]) & (df["volume"] >= 0)
    body_at_low = (df["open"] == df["close"]) & (df["close"] == df["low"])
    mask &= ~body_at_low
    return df[mask].copy()


def attach_cost_columns(
    df: pd.DataFrame, commission: float, spread: float, slippage: float
) -> pd.DataFrame:
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
    """Quantile-bucket future returns.

    ``pandas.qcut`` raises when it cannot build ``q`` unique bins (e.g. flat
    price series).  The stub pandas implementation shipped with the project has
    similar limitations, so we implement a small fallback that deterministically
    buckets the available distinct returns.
    """
    ret = _future_return(close, horizon)
    values = list(ret._values) if hasattr(ret, "_values") else list(ret)
    index = ret.index[:] if hasattr(ret, "index") else list(range(len(values)))

    valid_positions = [i for i, value in enumerate(values) if not pd.isna(value)]
    if not valid_positions:
        return pd.Series([math.nan] * len(values), index=index)

    unique_values = sorted({values[i] for i in valid_positions})
    if len(unique_values) == 1:
        result = [math.nan] * len(values)
        for pos in valid_positions:
            result[pos] = 0
        return pd.Series(result, index=index)

    bucket_count = min(q, len(unique_values))
    step = len(unique_values) / bucket_count
    bucket_map = {}
    for idx, value in enumerate(unique_values):
        bucket = min(bucket_count - 1, int(idx / step))
        bucket_map[value] = bucket

    result = [math.nan] * len(values)
    for pos in valid_positions:
        value = values[pos]
        result[pos] = bucket_map[value]
    return pd.Series(result, index=index)


def comp_m_scores(
    price_hist: Dict[str, np.ndarray], lookback: int = 60, skip: int = 0
) -> Dict[str, float]:
    """Cross-sectional momentum z-scores.

    Prices are expected as numpy arrays with the most recent bar at the end.  The
    trailing log return over ``lookback`` periods is computed for each asset,
    optionally skipping the last ``skip`` observations.  Assets with insufficient
    history are ignored.  The resulting scores are z-scored across the universe so
    that callers can compare momentum strength on a standardized scale.
    """

    scores: Dict[str, float] = {}
    vals: list[float] = []
    keys: list[str] = []

    for sym, px in price_hist.items():
        if px.size < lookback + skip + 1:
            continue
        segment = px[-(lookback + skip) : -skip] if skip > 0 else px[-lookback:]
        if segment.size == 0:
            continue
        ret = float(math.log(segment[-1] / segment[0]))
        vals.append(ret)
        keys.append(sym)

    if not vals:
        return scores

    arr = [float(v) for v in vals]
    mu = sum(arr) / len(arr)
    variance = sum((value - mu) ** 2 for value in arr) / len(arr)
    sd = math.sqrt(variance)
    if not math.isfinite(sd) or sd == 0.0:
        sd = 1e-12

    for key, value in zip(keys, arr):
        scores[key] = float((value - mu) / sd)
    return scores


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
        hit_upper = (
            (diff >= math.log(1 + upper)).idxmax() if (diff >= math.log(1 + upper)).any() else None
        )
        hit_lower = (
            (diff <= -math.log(1 + lower)).idxmax()
            if (diff <= -math.log(1 + lower)).any()
            else None
        )
        first_hit = None
        if hit_upper is not None:
            first_hit = hit_upper
            label = 1
        if hit_lower is not None and (
            first_hit is None or window.index.get_loc(hit_lower) < window.index.get_loc(first_hit)
        ):
            first_hit = hit_lower
            label = -1
        out.iloc[i] = label if first_hit is not None else 0
    return out


@dataclass
class FeatureStore:
    """Very small BigQuery-oriented feature store stub."""

    dataset: str = "market_fs"
    table: str = "features_ohlcv_min"

    def write(
        self,
        df: pd.DataFrame,
        feature_version: str,
        source_hash: str,
        *,
        project: Optional[str] = None,
    ) -> Tuple[str, str]:
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


def purged_kfold(
    n_splits: int, embargo: int, n_samples: int
) -> Iterator[Tuple[Iterable[int], Iterable[int]]]:
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
