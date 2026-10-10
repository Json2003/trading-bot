#!/usr/bin/env python3
"""Download completed Binance USD-M open-interest metrics for research."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://data.binance.vision"
FIELDS = ("timestamp", "sum_open_interest", "sum_open_interest_value")


def _days(since: str, until: str) -> list[datetime]:
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("--since must be earlier than exclusive --until")
    return [start + timedelta(days=i) for i in range((end - start).days)]


def _read(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "trading-bot-open-interest-research/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _parse_day(payload: bytes, symbol: str) -> tuple[list[dict[str, str]], int]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one metrics CSV for {symbol}, found {names}")
        with archive.open(names[0], "r") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            required = {"create_time", "symbol", "sum_open_interest", "sum_open_interest_value"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"metrics CSV missing columns for {symbol}")
            latest_by_hour: dict[str, dict[str, str]] = {}
            invalid_rows = 0
            for row in reader:
                if row["symbol"].upper() != symbol:
                    continue
                timestamp = datetime.strptime(
                    row["create_time"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                oi = float(row["sum_open_interest"])
                oi_value = float(row["sum_open_interest_value"])
                if (
                    not math.isfinite(oi)
                    or not math.isfinite(oi_value)
                    or oi <= 0
                    or oi_value <= 0
                ):
                    invalid_rows += 1
                    continue
                hour = timestamp.replace(minute=0, second=0, microsecond=0)
                latest_by_hour[hour.isoformat()] = {
                    "timestamp": hour.isoformat().replace("+00:00", "Z"),
                    "sum_open_interest": str(oi),
                    "sum_open_interest_value": str(oi_value),
                }
            return [latest_by_hour[key] for key in sorted(latest_by_hour)], invalid_rows


def fetch_symbol(symbol: str, since: str, until: str, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_timestamp: dict[str, dict[str, str]] = {}
    archives = []
    missing_dates = []
    invalid_row_count = 0
    for day in _days(since, until):
        date = day.strftime("%Y-%m-%d")
        filename = f"{symbol}-metrics-{date}.zip"
        url = f"{BASE_URL}/data/futures/um/daily/metrics/{symbol}/{filename}"
        try:
            payload = _read(url)
            checksum_text = _read(url + ".CHECKSUM").decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing_dates.append(date)
                continue
            raise
        expected = checksum_text.split()[0].lower()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch for {filename}")
        day_rows, invalid_rows = _parse_day(payload, symbol)
        invalid_row_count += invalid_rows
        for row in day_rows:
            previous = by_timestamp.get(row["timestamp"])
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate at {symbol} {row['timestamp']}")
            by_timestamp[row["timestamp"]] = row
        archives.append({"date": date, "url": url, "sha256": actual, "rows": len(day_rows), "invalid_rows_excluded": invalid_rows})
    rows = [by_timestamp[key] for key in sorted(by_timestamp)]
    path = output_dir / f"{symbol}_1h.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": BASE_URL,
        "market": "USD-M futures metrics",
        "symbol": symbol,
        "period": "5m archives downsampled to latest completed observation per hour",
        "since": since,
        "until_exclusive": until,
        "archive_count": len(archives),
        "missing_dates": missing_dates,
        "missing_day_count": len(missing_dates),
        "invalid_row_count_excluded": invalid_row_count,
        "row_count": len(rows),
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
        "archives": archives,
    }
    (output_dir / f"{symbol}_1h.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {"path": str(path), **{key: value for key, value in manifest.items() if key != "archives"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/historical/binance/open_interest"))
    args = parser.parse_args()
    result = {symbol: fetch_symbol(symbol, args.since, args.until, args.output_dir) for symbol in args.symbols}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
