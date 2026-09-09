#!/usr/bin/env python3
"""Merge verified daily increments into normalized historical hourly CSVs."""
from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path} must have columns {FIELDS}")
        rows = {}
        for row in reader:
            timestamp = row["timestamp"]
            if timestamp in rows and rows[timestamp] != row:
                raise ValueError(f"{path} contains conflicting timestamp {timestamp}")
            rows[timestamp] = {field: row[field] for field in FIELDS}
        return rows


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def validate_contiguous(rows: dict[str, dict[str, str]], symbol: str) -> None:
    timestamps = sorted(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in rows
    )
    for previous, current in zip(timestamps, timestamps[1:]):
        if (current - previous).total_seconds() > 5400:
            raise ValueError(
                f"{symbol} merged data has a gap from {previous.isoformat()} "
                f"to {current.isoformat()}"
            )


def merge_symbol(
    symbol: str,
    *,
    monthly_dir: Path,
    daily_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = read_csv(monthly_dir / f"{symbol}_1h.csv")
    daily_symbol_dir = daily_dir / symbol
    daily_paths = sorted(daily_symbol_dir.glob("*.csv"))
    for path in daily_paths:
        for timestamp, row in read_csv(path).items():
            existing = rows.get(timestamp)
            if existing is not None and existing != row:
                raise ValueError(f"{symbol} has conflicting monthly/daily candle {timestamp}")
            rows[timestamp] = row
    if not rows:
        raise ValueError(f"no data available for {symbol}")
    validate_contiguous(rows, symbol)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for timestamp in sorted(rows):
        writer.writerow(rows[timestamp])
    output_path = output_dir / f"{symbol}_1h.csv"
    atomic_write(output_path, output.getvalue())
    return {
        "symbol": symbol,
        "rows": len(rows),
        "from": min(rows),
        "through": max(rows),
        "daily_files_merged": len(daily_paths),
        "path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--monthly-dir", type=Path, default=Path("data/historical/binance/normalized"))
    parser.add_argument("--daily-dir", type=Path, default=Path("data/daily/binance"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/historical/binance/normalized"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/merged-data.json"))
    args = parser.parse_args()
    results = [
        merge_symbol(
            symbol,
            monthly_dir=args.monthly_dir,
            daily_dir=args.daily_dir,
            output_dir=args.output_dir,
        )
        for symbol in args.symbols
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_candles_only": True,
        "symbols": results,
    }
    atomic_write(args.output, json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
