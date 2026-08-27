#!/usr/bin/env python3
"""Append completed current-month Binance Vision spot candles and funding observations.

This is a research-only forward-data collector. Historical monthly archives remain
untouched; only current-month rows are appended after duplicate/conflict checks.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

VISION = "https://data.binance.vision"
FAPI = "https://fapi.binance.com"
FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "funding-8bps-repro/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def iso_ms(raw: str) -> str:
    value = int(float(raw))
    if value >= 10**14:
        value //= 1000
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def read_spot_archive(payload: bytes) -> list[dict[str, str]]:
    rows = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one spot CSV, found {names}")
        with archive.open(names[0], "r") as raw:
            for values in csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline="")):
                if not values or values[0].lower() in {"open time", "open_time"}:
                    continue
                if len(values) < 6:
                    raise ValueError("spot row has fewer than six columns")
                rows.append({
                    "timestamp": iso_ms(values[0]),
                    "open": values[1],
                    "high": values[2],
                    "low": values[3],
                    "close": values[4],
                    "volume": values[5],
                })
    return rows


def merge_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = {row["timestamp"]: row for row in csv.DictReader(handle)}
    for row in rows:
        prior = existing.get(row["timestamp"])
        if prior is not None and any(prior.get(key) != row.get(key) for key in fields[1:]):
            raise ValueError(f"conflicting duplicate at {path} {row['timestamp']}")
        existing[row["timestamp"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing[key] for key in sorted(existing))


def update_spot(symbol: str, cutoff: datetime, out_dir: Path) -> dict:
    first = cutoff.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    day = first
    rows = []
    # A daily archive is used only after its UTC day is complete.
    last_day = cutoff.date() - timedelta(days=1)
    while day.date() <= last_day:
        stem = f"{symbol}-1h-{day:%Y-%m-%d}"
        url = f"{VISION}/data/spot/daily/klines/{symbol}/1h/{stem}.zip"
        try:
            rows.extend(read_spot_archive(get(url)))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        day += timedelta(days=1)
    path = out_dir / f"{symbol}_1h.csv"
    merge_csv(path, rows, FIELDS)
    return {"symbol": symbol, "new_rows_seen": len(rows), "path": str(path)}


def update_funding(symbol: str, cutoff: datetime, out_dir: Path) -> dict:
    path = out_dir / f"{symbol}_funding.csv"
    rows = []
    start_ms = int(cutoff.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    end_ms = int(cutoff.timestamp() * 1000)
    url = f"{FAPI}/fapi/v1/fundingRate?{urllib.parse.urlencode({'symbol': symbol, 'startTime': start_ms, 'endTime': end_ms, 'limit': 1000})}"
    payload = json.loads(get(url))
    if isinstance(payload, dict) and payload.get("code"):
        raise RuntimeError(f"funding endpoint error: {payload}")
    for item in payload:
        rate_time = datetime.fromtimestamp(int(item["fundingTime"]) / 1000, timezone.utc)
        rows.append({
            "timestamp": rate_time.isoformat().replace("+00:00", "Z"),
            "funding_rate": str(item["fundingRate"]),
            "interval_hours": "8",
        })
    merge_csv(path, rows, ["timestamp", "funding_rate", "interval_hours"])
    return {"symbol": symbol, "new_rows_seen": len(rows), "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True, help="exclusive UTC cutoff, ISO-8601")
    parser.add_argument("--spot-dir", type=Path, default=Path("data/historical/binance/normalized"))
    parser.add_argument("--funding-dir", type=Path, default=Path("data/historical/binance/funding"))
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
    result = {
        "cutoff": cutoff.isoformat(),
        "spot": [update_spot(symbol, cutoff, args.spot_dir) for symbol in ("BTCUSDT", "ETHUSDT")],
        "funding": [update_funding(symbol, cutoff, args.funding_dir) for symbol in ("BTCUSDT", "ETHUSDT")],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
