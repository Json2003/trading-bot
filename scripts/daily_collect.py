#!/usr/bin/env python3
"""Fetch and verify completed Binance spot hourly archives.

This collector is data-only. It never calls an exchange trading endpoint,
requires no credentials, and refuses incomplete, malformed, duplicate, or
gapped UTC days. Existing verified day files are reused.
"""
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

INTERVAL = "1h"
FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "trading-bot-daily-data/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def checksum(zip_bytes: bytes, checksum_text: str) -> str:
    expected = checksum_text.strip().split()[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"unrecognized checksum: {checksum_text!r}")
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    return actual


def timestamp_iso(raw: str) -> str:
    value = int(float(raw))
    if value >= 10**14:
        value //= 1000
    return (
        datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_archive(zip_bytes: bytes) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in archive, found {names}")
        rows: list[dict[str, str]] = []
        with archive.open(names[0], "r") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for values in reader:
                if not values or values[0].lower() in {"open time", "open_time"}:
                    continue
                if len(values) < 6:
                    raise ValueError("kline row has fewer than six columns")
                rows.append(
                    {
                        "timestamp": timestamp_iso(values[0]),
                        "open": values[1],
                        "high": values[2],
                        "low": values[3],
                        "close": values[4],
                        "volume": values[5],
                    }
                )
    return rows


def validate_rows(rows: list[dict[str, str]], target: date) -> dict[str, Any]:
    if len(rows) != 24:
        raise ValueError(f"{target} must contain 24 completed hourly candles, got {len(rows)}")
    rows.sort(key=lambda row: row["timestamp"])
    expected = [
        datetime(
            target.year,
            target.month,
            target.day,
            hour,
            tzinfo=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
        for hour in range(24)
    ]
    timestamps = [row["timestamp"] for row in rows]
    if timestamps != expected or len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{target} is not a complete contiguous UTC day")
    for row in rows:
        values = [float(row[field]) for field in FIELDS[1:]]
        open_price, high, low, close, volume = values
        if (
            not all(math.isfinite(value) for value in values)
            or min(open_price, high, low, close) <= 0
            or high < max(open_price, close)
            or low > min(open_price, close)
            or volume < 0
        ):
            raise ValueError(f"{target} contains invalid OHLCV values")
    return {
        "rows": len(rows),
        "from": timestamps[0],
        "through": timestamps[-1],
    }


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def collect_day(
    symbol: str,
    target: date,
    *,
    out_dir: Path,
    base_url: str,
) -> dict[str, Any]:
    symbol_dir = out_dir / symbol
    day_path = symbol_dir / f"{target.isoformat()}.csv"
    manifest_path = symbol_dir / f"{target.isoformat()}.json"
    if day_path.is_file() and manifest_path.is_file():
        return {
            "symbol": symbol,
            "date": target.isoformat(),
            "path": str(day_path),
            "status": "reused",
        }

    stem = f"{symbol}-{INTERVAL}-{target.isoformat()}"
    url = f"{base_url}/data/spot/daily/klines/{symbol}/{INTERVAL}/{stem}.zip"
    try:
        zip_bytes = download(url)
        checksum_text = download(f"{url}.CHECKSUM").decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"completed archive unavailable for {symbol} {target.isoformat()}: HTTP {exc.code}"
        ) from exc

    digest = checksum(zip_bytes, checksum_text)
    rows = read_archive(zip_bytes)
    quality = validate_rows(rows, target)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(day_path, output.getvalue())
    atomic_write(
        manifest_path,
        json.dumps(
            {
                "schema_version": 1,
                "symbol": symbol,
                "date": target.isoformat(),
                "interval": INTERVAL,
                "source": url,
                "archive_sha256": digest,
                "completed_candles_only": True,
                **quality,
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return {
        "symbol": symbol,
        "date": target.isoformat(),
        "path": str(day_path),
        "status": "downloaded",
        **quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument(
        "--days",
        type=int,
        default=45,
        help="number of completed UTC days ending yesterday to verify",
    )
    parser.add_argument("--date", type=lambda value: date.fromisoformat(value))
    parser.add_argument("--out-dir", type=Path, default=Path("data/daily/binance"))
    parser.add_argument("--base-url", default="https://data.binance.vision")
    args = parser.parse_args()
    if args.days < 1 or args.days > 366:
        raise ValueError("--days must be between 1 and 366")
    end = args.date or (datetime.now(timezone.utc).date() - timedelta(days=1))
    if end >= datetime.now(timezone.utc).date():
        raise ValueError("--date must be a completed UTC day")
    targets = [end - timedelta(days=offset) for offset in range(args.days - 1, -1, -1)]
    results = [
        collect_day(symbol, target, out_dir=args.out_dir, base_url=args.base_url)
        for symbol in args.symbols
        for target in targets
    ]
    index = {
        "schema_version": 1,
        "source": args.base_url,
        "interval": INTERVAL,
        "symbols": sorted(args.symbols),
        "completed_candles_only": True,
        "days_requested": args.days,
        "through": end.isoformat(),
        "files": results,
    }
    atomic_write(args.out_dir / "index.json", json.dumps(index, indent=2, sort_keys=True))
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
