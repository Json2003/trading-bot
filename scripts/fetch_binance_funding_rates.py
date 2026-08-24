#!/usr/bin/env python3
"""Download completed Binance USD-M funding-rate archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://data.binance.vision"


def months(since: str, until: str) -> list[tuple[int, int]]:
    start = datetime.strptime(since, "%Y-%m")
    end = datetime.strptime(until, "%Y-%m")
    out = []
    year, month = start.year, start.month
    while (year, month) < (end.year, end.month):
        out.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def fetch(symbol: str, since: str, until: str, output: Path) -> dict:
    rows = {}
    archives = []
    for year, month in months(since, until):
        stem = f"{symbol}-fundingRate-{year:04d}-{month:02d}"
        url = f"{BASE}/data/futures/um/monthly/fundingRate/{symbol}/{stem}.zip"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"expected one funding CSV for {symbol} {year}-{month}")
            for row in csv.DictReader(io.TextIOWrapper(archive.open(names[0]), encoding="utf-8")):
                timestamp = datetime.fromtimestamp(
                    int(float(row["calc_time"])) / 1000, tz=timezone.utc
                ).isoformat()
                rows[timestamp] = {
                    "timestamp": timestamp,
                    "funding_rate": float(row["last_funding_rate"]),
                    "interval_hours": int(row["funding_interval_hours"]),
                }
        archives.append({"url": url, "sha256": digest, "rows": len(rows)})
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "funding_rate", "interval_hours"])
        writer.writeheader()
        writer.writerows(ordered)
    return {
        "symbol": symbol,
        "rows": len(ordered),
        "start": ordered[0]["timestamp"] if ordered else None,
        "end": ordered[-1]["timestamp"] if ordered else None,
        "archives": archives,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/historical/binance/funding"))
    args = parser.parse_args()
    result = {
        symbol: fetch(symbol, args.since, args.until, args.output_dir / f"{symbol}_funding.csv")
        for symbol in args.symbols
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
