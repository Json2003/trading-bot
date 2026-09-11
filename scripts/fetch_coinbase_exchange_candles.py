#!/usr/bin/env python3
"""Download completed Coinbase Exchange hourly candles for research.

Coinbase's public candles endpoint returns at most 300 candles per request.
This downloader paginates fixed completed-hour windows, preserves gaps in the
manifest, and never fills missing candles with synthetic values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_SECONDS = 3600
MAX_CANDLES_PER_REQUEST = 250
FIELDNAMES = ("timestamp", "open", "high", "low", "close", "volume")


def _utc_date(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _download_json(url: str) -> list[list[object]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "trading-bot-cross-exchange-research/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Coinbase candle request failed: HTTP {exc.code}: {url}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Coinbase candle response was not a list: {payload!r}")
    return payload


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_rows(payload: list[list[object]], product: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for values in payload:
        if len(values) < 6:
            raise ValueError(f"{product} candle row has fewer than six fields")
        timestamp = int(values[0])
        low, high, open_price, close, volume = (float(value) for value in values[1:6])
        if (
            timestamp <= 0
            or not all(math.isfinite(value) for value in (low, high, open_price, close, volume))
            or min(low, high, open_price, close) <= 0
            or high < max(open_price, close)
            or low > min(open_price, close)
            or volume < 0
        ):
            raise ValueError(f"invalid Coinbase candle for {product}: {values!r}")
        rows.append(
            {
                "timestamp": _iso(timestamp),
                "open": str(open_price),
                "high": str(high),
                "low": str(low),
                "close": str(close),
                "volume": str(volume),
            }
        )
    return rows


def fetch_product(product: str, since: str, until: str, output_dir: Path) -> dict[str, object]:
    start = _utc_date(since).replace(minute=0, second=0, microsecond=0)
    end = _utc_date(until).replace(minute=0, second=0, microsecond=0)
    if start >= end:
        raise ValueError("--since must be earlier than exclusive --until")

    rows_by_timestamp: dict[str, dict[str, str]] = {}
    chunks: list[dict[str, object]] = []
    current = start
    while current < end:
        chunk_end = min(
            end,
            current + timedelta(hours=MAX_CANDLES_PER_REQUEST),
        )
        query = urllib.parse.urlencode(
            {
                "start": current.isoformat().replace("+00:00", "Z"),
                "end": chunk_end.isoformat().replace("+00:00", "Z"),
                "granularity": GRANULARITY_SECONDS,
            }
        )
        url = f"{BASE_URL}/products/{product}/candles?{query}"
        parsed = _parse_rows(_download_json(url), product)
        for row in parsed:
            timestamp = row["timestamp"]
            existing = rows_by_timestamp.get(timestamp)
            if existing is not None and existing != row:
                raise ValueError(f"conflicting duplicate Coinbase candle at {product} {timestamp}")
            rows_by_timestamp[timestamp] = row
        chunks.append({"start": current.isoformat(), "end_exclusive": chunk_end.isoformat(), "rows": len(parsed)})
        current = chunk_end
        time.sleep(0.12)

    rows = [
        row for timestamp, row in sorted(rows_by_timestamp.items())
        if start <= _utc_date(timestamp) < end
    ]
    if not rows:
        raise ValueError(f"Coinbase returned no candles for {product}")

    timestamps = [_utc_date(row["timestamp"]) for row in rows]
    gaps = [
        {
            "from": previous.isoformat(),
            "to": current_timestamp.isoformat(),
            "missing_hours": int((current_timestamp - previous).total_seconds() / 3600) - 1,
        }
        for previous, current_timestamp in zip(timestamps, timestamps[1:])
        if (current_timestamp - previous).total_seconds() > 5400
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{product.replace('-', '')}_1h.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source": BASE_URL,
        "product": product,
        "granularity_seconds": GRANULARITY_SECONDS,
        "since": since,
        "until_exclusive": until,
        "request_count": len(chunks),
        "row_count": len(rows),
        "start": rows[0]["timestamp"],
        "end": rows[-1]["timestamp"],
        "gaps_over_90_minutes": gaps,
        "chunks": chunks,
    }
    manifest_path = output_dir / f"{product.replace('-', '')}_1h.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", nargs="+", default=["BTC-USD", "ETH-USD"])
    parser.add_argument("--since", required=True, help="inclusive UTC timestamp")
    parser.add_argument("--until", required=True, help="exclusive UTC timestamp")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/historical/coinbase/normalized"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                product: fetch_product(product, args.since, args.until, args.output_dir)
                for product in args.products
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
