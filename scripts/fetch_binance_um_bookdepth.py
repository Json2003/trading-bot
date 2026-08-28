#!/usr/bin/env python3
"""Download Binance USD-M daily book-depth archives and normalize top-1% imbalance.

The downloader is research-only. Missing archives remain missing in the manifest;
they are never converted into zero order-book pressure.
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
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://data.binance.vision"
FIELDNAMES = ("timestamp", "imbalance", "snapshot_count", "bid_notional", "ask_notional")


def day_iter(since: str, until: str) -> list[datetime]:
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("--since must be earlier than exclusive --until")
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "trading-bot-book-depth-research/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _checksum(zip_bytes: bytes, checksum_text: str) -> str:
    expected = checksum_text.strip().split()[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"unrecognized checksum format: {checksum_text!r}")
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    return actual


def _timestamp(raw: str) -> datetime:
    value = raw.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_archive(zip_bytes: bytes) -> list[dict[str, object]]:
    snapshots: dict[datetime, dict[int, float]] = defaultdict(dict)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in archive, found {names}")
        with archive.open(names[0], "r") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            required = {"timestamp", "percentage", "notional"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(
                    f"book-depth CSV missing columns: {sorted(required - set(reader.fieldnames or []))}"
                )
            for row in reader:
                percentage = int(row["percentage"])
                if percentage not in {-1, 1}:
                    continue
                notional = float(row["notional"])
                if not math.isfinite(notional) or notional < 0:
                    raise ValueError("book-depth notional must be finite and non-negative")
                snapshots[_timestamp(row["timestamp"])][percentage] = notional

    return [
        {
            "timestamp": timestamp,
            "imbalance": (levels[-1] - levels[1]) / (levels[-1] + levels[1]),
            "bid_notional": levels[-1],
            "ask_notional": levels[1],
        }
        for timestamp, levels in snapshots.items()
        if -1 in levels and 1 in levels and levels[-1] + levels[1] > 0
    ]


def fetch_symbol(symbol: str, since: str, until: str, raw_dir: Path, out_dir: Path) -> dict[str, object]:
    rows_by_hour: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    archives: list[dict[str, object]] = []
    missing_dates: list[str] = []
    raw_symbol_dir = raw_dir / symbol
    raw_symbol_dir.mkdir(parents=True, exist_ok=True)

    for day in day_iter(since, until):
        date = day.strftime("%Y-%m-%d")
        stem = f"{symbol}-bookDepth-{date}"
        url = f"{BASE_URL}/data/futures/um/daily/bookDepth/{symbol}/{stem}.zip"
        checksum_url = f"{url}.CHECKSUM"
        try:
            zip_bytes = _download(url)
            checksum_text = _download(checksum_url).decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing_dates.append(date)
                continue
            raise RuntimeError(
                f"book-depth archive request failed for {symbol} {date}: HTTP {exc.code}"
            ) from exc

        digest = _checksum(zip_bytes, checksum_text)
        day_rows = _read_archive(zip_bytes)
        for row in day_rows:
            timestamp = row["timestamp"]
            hour = timestamp.replace(minute=0, second=0, microsecond=0)
            rows_by_hour[hour].append(row)
        archives.append({"url": url, "sha256": digest, "snapshots": len(day_rows)})

    rows: list[dict[str, object]] = []
    for hour in sorted(rows_by_hour):
        group = rows_by_hour[hour]
        rows.append(
            {
                "timestamp": hour.isoformat().replace("+00:00", "Z"),
                "imbalance": sum(float(row["imbalance"]) for row in group) / len(group),
                "snapshot_count": len(group),
                "bid_notional": sum(float(row["bid_notional"]) for row in group) / len(group),
                "ask_notional": sum(float(row["ask_notional"]) for row in group) / len(group),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{symbol}_bookdepth_1h.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source": BASE_URL,
        "market": "USD-M futures daily bookDepth archives",
        "symbol": symbol,
        "since": since,
        "until_exclusive": until,
        "band_percentage": 1,
        "aggregation": "mean of complete +/-1 percent snapshots within each UTC hour",
        "archive_count": len(archives),
        "missing_dates": missing_dates,
        "missing_day_count": len(missing_dates),
        "hour_count": len(rows),
        "snapshot_count": sum(int(row["snapshot_count"]) for row in rows),
        "archives": archives,
    }
    (out_dir / f"{symbol}_bookdepth_1h.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {key: value for key, value in manifest.items() if key != "archives"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--since", required=True, help="inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="exclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/historical/binance/raw_bookdepth"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/historical/binance/bookdepth"))
    args = parser.parse_args()
    print(
        json.dumps(
            {
                symbol: fetch_symbol(symbol, args.since, args.until, args.raw_dir, args.out_dir)
                for symbol in args.symbols
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
