#!/usr/bin/env python3
"""Download completed recent Binance spot hourly candles from daily archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE_URL = "https://data.binance.vision"
FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


def _days(since: str, until: str) -> list[datetime]:
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("--since must be earlier than exclusive --until")
    return [start + timedelta(days=i) for i in range((end - start).days)]


def _read(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "trading-bot-recent-candles/1.0"}
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            last_error = exc
        if attempt < 4:
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _timestamp(raw: str) -> datetime:
    value = int(float(raw))
    if value >= 10**14:
        value //= 1000
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _read_archive(payload: bytes) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV, found {names}")
        rows: list[dict[str, str]] = []
        with archive.open(names[0], "r") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for values in reader:
                if not values or values[0].lower() in {"open time", "open_time"}:
                    continue
                if len(values) < 6:
                    continue
                timestamp = _timestamp(values[0])
                values_float = [float(value) for value in values[1:6]]
                if (
                    not all(math.isfinite(value) for value in values_float)
                    or min(values_float[:4]) <= 0
                    or values_float[4] < 0
                ):
                    continue
                rows.append(
                    {
                        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                        "open": values[1],
                        "high": values[2],
                        "low": values[3],
                        "close": values[4],
                        "volume": values[5],
                    }
                )
        return rows


def fetch_symbol(
    symbol: str,
    since: str,
    until: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_1h.csv"
    manifest_path = output_dir / f"{symbol}_1h.manifest.json"
    by_timestamp: dict[str, dict[str, str]] = {}
    archives: list[dict[str, Any]] = []
    missing_dates: list[str] = []

    for day in _days(since, until):
        date = day.strftime("%Y-%m-%d")
        stem = f"{symbol}-1h-{date}"
        url = f"{BASE_URL}/data/spot/daily/klines/{symbol}/1h/{stem}.zip"
        try:
            payload = _read(url)
            checksum_text = _read(url + ".CHECKSUM").decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing_dates.append(date)
                continue
            raise
        expected = checksum_text.strip().split()[0].lower()
        actual = hashlib.sha256(payload).hexdigest()
        if expected != actual:
            raise ValueError(f"checksum mismatch for {stem}")
        day_rows = _read_archive(payload)
        for row in day_rows:
            previous = by_timestamp.get(row["timestamp"])
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate at {symbol} {row['timestamp']}")
            by_timestamp[row["timestamp"]] = row
        archives.append(
            {
                "date": date,
                "url": url,
                "sha256": actual,
                "rows": len(day_rows),
            }
        )

    rows = [by_timestamp[key] for key in sorted(by_timestamp)]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source": BASE_URL,
        "market": "spot daily 1h klines",
        "symbol": symbol,
        "interval": "1h",
        "requested_start": since,
        "requested_end_exclusive": until,
        "archive_count": len(archives),
        "missing_dates": missing_dates,
        "complete": not missing_dates,
        "row_count": len(rows),
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
        "archives": archives,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {
        symbol: fetch_symbol(symbol, args.since, args.until, args.output_dir)
        for symbol in args.symbols
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
