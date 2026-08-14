#!/usr/bin/env python3
"""Download and normalize Binance Vision spot kline archives.

The output is deliberately a simple CSV accepted by
``scripts/run_historical_momentum_backtest.py``.  Binance's public archive is
used instead of an exchange API so a backtest can be reproduced later without
API keys or exchange pagination changes.

Example:
    python scripts/fetch_binance_vision_klines.py \
        --symbols BTCUSDT ETHUSDT --since 2023-01 --until 2026-08

``--until`` is exclusive and uses UTC calendar months.  Raw archives and their
checksums are retained by default; do not commit either directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://data.binance.vision"
INTERVAL = "1h"
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
)


def month_iter(since: str, until: str) -> list[tuple[int, int]]:
    start = datetime.strptime(since, "%Y-%m")
    end = datetime.strptime(until, "%Y-%m")
    if start >= end:
        raise ValueError("--since must be earlier than exclusive --until")
    result: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) < (end.year, end.month):
        result.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "trading-bot-historical-data/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _checksum(zip_bytes: bytes, checksum_text: str) -> str:
    expected = checksum_text.strip().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"unrecognized checksum format: {checksum_text!r}")
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    return actual


def _timestamp_iso(raw: str) -> str:
    value = int(float(raw))
    # Binance spot archive timestamps are milliseconds before 2025 and
    # microseconds from 2025 onward. Normalize both to UTC milliseconds.
    if value >= 10**14:
        value //= 1000
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _read_archive(zip_bytes: bytes) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV in archive, found {csv_names}")
        with archive.open(csv_names[0], "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text)
            for values in reader:
                if not values or values[0].lower() in {"open time", "open_time"}:
                    continue
                if len(values) < 6:
                    raise ValueError("kline row has fewer than six columns")
                rows.append({
                    "timestamp": _timestamp_iso(values[0]),
                    "open": values[1],
                    "high": values[2],
                    "low": values[3],
                    "close": values[4],
                    "volume": values[5],
                })
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        raise ValueError("no kline rows found")
    rows.sort(key=lambda row: row["timestamp"])
    timestamps = [datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) for row in rows]
    duplicate_count = len(timestamps) - len(set(timestamps))
    if duplicate_count:
        raise ValueError(f"normalized data contains {duplicate_count} duplicate timestamps")
    invalid = []
    for row in rows:
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        if min(high, low, close) <= 0 or high < low:
            invalid.append(row["timestamp"])
    if invalid:
        raise ValueError(f"invalid OHLC values at {invalid[:3]}")
    gaps = []
    for previous, current in zip(timestamps, timestamps[1:]):
        hours = (current - previous).total_seconds() / 3600
        if hours > 1.5:
            gaps.append({"from": previous.isoformat(), "to": current.isoformat(), "hours": hours})
    return {
        "rows": len(rows),
        "start": rows[0]["timestamp"],
        "end": rows[-1]["timestamp"],
        "gaps_over_1_5_hours": gaps,
    }


def fetch_symbol(symbol: str, since: str, until: str, raw_dir: Path, out_dir: Path) -> dict[str, object]:
    rows_by_timestamp: dict[str, dict[str, str]] = {}
    archives: list[dict[str, object]] = []
    raw_symbol_dir = raw_dir / symbol
    raw_symbol_dir.mkdir(parents=True, exist_ok=True)

    for year, month in month_iter(since, until):
        stem = f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}"
        zip_path = raw_symbol_dir / f"{stem}.zip"
        checksum_path = raw_symbol_dir / f"{stem}.zip.CHECKSUM"
        url = f"{BASE_URL}/data/spot/monthly/klines/{symbol}/{INTERVAL}/{stem}.zip"
        try:
            zip_bytes = zip_path.read_bytes() if zip_path.exists() else _download(url)
            if not zip_path.exists():
                zip_path.write_bytes(zip_bytes)
            checksum_url = f"{url}.CHECKSUM"
            checksum_text = checksum_path.read_text(encoding="utf-8") if checksum_path.exists() else _download(checksum_url).decode()
            checksum_path.write_text(checksum_text, encoding="utf-8")
            digest = _checksum(zip_bytes, checksum_text)
            month_rows = _read_archive(zip_bytes)
            for row in month_rows:
                rows_by_timestamp[row["timestamp"]] = row
            archives.append({"file": zip_path.name, "url": url, "sha256": digest, "rows": len(month_rows)})
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"archive unavailable for {symbol} {year:04d}-{month:02d}: HTTP {exc.code}; use a completed month") from exc

    rows = list(rows_by_timestamp.values())
    summary = _validate_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{symbol}_{INTERVAL}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["timestamp"]))
    manifest = {"source": BASE_URL, "symbol": symbol, "interval": INTERVAL, "since": since, "until_exclusive": until, "archives": archives, **summary}
    (out_dir / f"{symbol}_{INTERVAL}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output": str(output_path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--since", required=True, help="inclusive UTC month, YYYY-MM")
    parser.add_argument("--until", required=True, help="exclusive UTC month, YYYY-MM")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/historical/binance/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/historical/binance/normalized"))
    args = parser.parse_args()
    print(json.dumps({symbol: fetch_symbol(symbol, args.since, args.until, args.raw_dir, args.out_dir) for symbol in args.symbols}, indent=2))


if __name__ == "__main__":
    main()
