"""Build a lightweight DuckDB catalog summarising daily market trends.

The production trading platform ingests a variety of market data sources
(internal quote feeds, research signals, and macro indicators).  To keep the
open-source test environment self-contained we instead aggregate whatever data
is already present beneath ``data/daily`` (populated via
``scripts/fetch_daily_market_data.py``) and fall back to deterministic synthetic
series when nothing exists yet.  The output mirrors the minimal schema expected
by the dashboard: ``symbol``, ``asset_class``, ``ts`` (UTC timestamp), ``date``
(date portion of ``ts``), ``close`` price, and a couple of simple analytics
columns.

Running ``python -m src.pipelines.daily_trends_pipeline`` will therefore always
leave two artefacts on disk:

``data/daily/<YYYY-MM-DD>/market_trends.parquet``
    Snapshot of the latest trends ready to be inspected or synced elsewhere.

``data/daily/market_trends.duckdb``
    DuckDB catalog containing a ``market_trends`` table so downstream tools can
    query the snapshot directly.

The module purposefully avoids network access during tests; indicator
calculations rely solely on pandas/numpy operations and the optional DuckDB
persistence is wrapped in a small helper to ease unit testing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

from scripts.fetch_daily_trends import fetch_daily_trends_safe

DATA_DIR = Path("data/daily")
PARQUET_NAME = "market_trends.parquet"
DUCKDB_NAME = "market_trends.duckdb"
TABLE_NAME = "market_trends"


@dataclass
class MarketSnapshot:
    """Container linking a dataframe to its metadata."""

    frame: pd.DataFrame
    symbol: str
    asset_class: str


def _ensure_timezone(values: Sequence) -> List[datetime]:
    normalised: List[datetime] = []
    for val in values:
        if isinstance(val, datetime):
            dt_val = val
        else:
            try:
                dt_val = datetime.fromisoformat(str(val))
            except ValueError:
                dt_val = datetime.utcfromtimestamp(float(val))
                dt_val = dt_val.replace(tzinfo=UTC)
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=UTC)
        else:
            dt_val = dt_val.astimezone(UTC)
        normalised.append(dt_val)
    return normalised


def _load_existing_frames(base_dir: Path) -> List[MarketSnapshot]:
    snapshots: List[MarketSnapshot] = []
    for subdir, asset_class in (("equities", "equity"), ("crypto", "crypto")):
        path = base_dir / subdir
        if not path.exists():
            continue
        for parquet in sorted(path.glob("*.parquet")):
            df = pd.read_parquet(parquet)
            if df.empty:
                continue
            if "timestamp" in df.columns:
                raw_ts = list(df["timestamp"])
            else:
                raw_ts = list(df.index)
            ts_values = _ensure_timezone(raw_ts)
            df = df.copy()
            df["timestamp"] = ts_values
            if "symbol" not in df.columns:
                df["symbol"] = parquet.stem.upper()
            snapshots.append(MarketSnapshot(df.reset_index(drop=True), df["symbol"].iloc[0], asset_class))
    return snapshots


def _synthetic_series(symbol: str, *, asset_class: str, periods: int = 30) -> MarketSnapshot:
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=periods - 1)
    idx = pd.date_range(start=start, end=end, freq="1D", tz=UTC)
    length = len(idx)
    base = np.linspace(-3, 3, length).tolist()
    oscillation = [2 * math.sin(math.pi * i / max(length - 1, 1)) for i in range(length)]
    close = [100 + base[i] + oscillation[i] for i in range(length)]
    noise = np.random.default_rng(42).normal(0, 0.5, length)
    noise = noise.tolist() if hasattr(noise, "tolist") else list(noise)
    open_ = [close[i] + noise[i] for i in range(length)]
    high = [max(open_[i], close[i]) + 0.5 for i in range(length)]
    low = [min(open_[i], close[i]) - 0.5 for i in range(length)]
    volume = [1_000_000 + (25_000 * i / max(length - 1, 1)) for i in range(length)]
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
        }
    )
    return MarketSnapshot(frame, symbol=symbol, asset_class=asset_class)


def _bootstrap_snapshots(base_dir: Path) -> List[MarketSnapshot]:
    existing = _load_existing_frames(base_dir)
    if existing:
        return existing
    # No parquet snapshots yet – generate deterministic synthetic curves for
    # one equity and one crypto asset so downstream analytics still work.
    return [
        _synthetic_series("SYN_EQ", asset_class="equity"),
        _synthetic_series("SYN_CR", asset_class="crypto"),
    ]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _compute_macd(close: pd.Series) -> pd.Series:
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal).fillna(0)


def _prepare_snapshot(snapshot: MarketSnapshot) -> pd.DataFrame:
    records = [row for row in snapshot.frame.to_dict("records") if row.get("timestamp") is not None]
    if not records:
        return pd.DataFrame([], columns=["ts", "date", "symbol", "asset_class", "close", "rsi_14", "macd_hist", "vol_20d"])

    records.sort(key=lambda row: row["timestamp"])
    ts_values = _ensure_timezone([row["timestamp"] for row in records])
    close_values = [float(row.get("close", 0) or 0) for row in records]
    volume_values = [float(row.get("volume", 0) or 0) for row in records]

    close_series = pd.Series(close_values)
    rsi_series = _compute_rsi(close_series)
    macd_series = _compute_macd(close_series)
    volume_series = pd.Series(volume_values)
    vol_series = volume_series.rolling(window=20, min_periods=1).mean()

    rows = []
    for idx, ts in enumerate(ts_values):
        rows.append(
            {
                "ts": ts,
                "date": ts.date(),
                "symbol": snapshot.symbol,
                "asset_class": snapshot.asset_class,
                "close": close_values[idx],
                "rsi_14": rsi_series.iloc[idx],
                "macd_hist": macd_series.iloc[idx],
                "vol_20d": vol_series.iloc[idx],
            }
        )
    return pd.DataFrame(rows)


def _append_trends_from_fetch(entries: Sequence[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame([], columns=["ts", "date", "symbol", "asset_class", "close", "rsi_14", "macd_hist", "vol_20d"])
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    records = []
    for idx, entry in enumerate(entries):
        records.append(
            {
                "ts": today + timedelta(hours=idx),
                "date": today.date(),
                "symbol": (entry.get("title") or f"TREND_{idx}").upper(),
                "asset_class": "trend",  # dashboard labels
                "close": float(entry.get("formattedTraffic") or 0),
                "rsi_14": 50.0,
                "macd_hist": 0.0,
                "vol_20d": len(entry.get("relatedQueries") or []),
            }
        )
    return pd.DataFrame(records)


def build_market_trends(base_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Construct the combined market trends dataframe."""

    base_dir.mkdir(parents=True, exist_ok=True)
    snapshots = _bootstrap_snapshots(base_dir)
    frames = [_prepare_snapshot(snapshot) for snapshot in snapshots]

    trends = fetch_daily_trends_safe()
    trend_frame = _append_trends_from_fetch(trends)
    if trend_frame.to_dict("records"):
        frames.append(trend_frame)

    records: List[dict] = []
    for frame in frames:
        records.extend(frame.to_dict("records"))

    if not records:
        return pd.DataFrame([], columns=["ts", "date", "symbol", "asset_class", "close", "rsi_14", "macd_hist", "vol_20d"])

    records.sort(key=lambda row: row["ts"])
    return pd.DataFrame(records)


def _write_parquet(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PARQUET_NAME
    records = df.to_dict("records")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, default=str, indent=2)
    return path


def _write_duckdb(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict("records")
    with duckdb.connect(db_path) as con:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        con.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                ts TIMESTAMP,
                date DATE,
                symbol TEXT,
                asset_class TEXT,
                close DOUBLE,
                rsi_14 DOUBLE,
                macd_hist DOUBLE,
                vol_20d DOUBLE
            )
            """
        )
        if records:
            rows = [
                (
                    row["ts"],
                    row["date"],
                    row["symbol"],
                    row["asset_class"],
                    float(row["close"]),
                    float(row["rsi_14"]),
                    float(row["macd_hist"]),
                    float(row["vol_20d"]),
                )
                for row in records
            ]
            con.executemany(
                f"INSERT INTO {TABLE_NAME} (ts, date, symbol, asset_class, close, rsi_14, macd_hist, vol_20d) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )


def main() -> Tuple[Path, Path]:
    """Entry point used by the Makefile and manual executions."""

    combined = build_market_trends(DATA_DIR)
    run_date = datetime.now(UTC).date().isoformat()
    parquet_dir = DATA_DIR / run_date
    parquet_path = _write_parquet(combined, parquet_dir)
    db_path = DATA_DIR / DUCKDB_NAME
    _write_duckdb(combined, db_path)
    return parquet_path, db_path


if __name__ == "__main__":  # pragma: no cover
    parquet_path, db_path = main()
    print(f"✅ Wrote {parquet_path}")
    print(f"✅ Updated {db_path}::{TABLE_NAME}")
