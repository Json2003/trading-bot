#!/usr/bin/env python3
"""Download the current Binance top-trader positioning history.

Binance exposes this series only as a rolling recent window. The public market-data collector
never treats a missing key, unavailable endpoint, or missing observation as a
zero signal; it writes an explicit blocked manifest so the research runner can
record a skip instead of manufacturing evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_URLS = ("https://fapi.binance.com", "https://www.binance.com")
ENDPOINTS = {
    "account": "/futures/data/topLongShortAccountRatio",
    "position": "/futures/data/topLongShortPositionRatio",
}
FIELDS = (
    "timestamp",
    "account_long",
    "account_short",
    "position_long",
    "position_short",
    "account_long_short_ratio",
    "position_long_short_ratio",
)
MAX_LIMIT = 500


class DataUnavailable(RuntimeError):
    """A source was unavailable; this is recorded as a skip, not a result."""


def _day_start(raw: str) -> datetime:
    value = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return value


def _request(
    base_url: str,
    endpoint: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    api_key: str,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "period": "1h",
            "limit": MAX_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
    )
    request = urllib.request.Request(
        f"{base_url}{endpoint}?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "trading-bot-large-trader-research/1.0",
        },
    )
    if api_key:
        request.add_header("X-MBX-APIKEY", api_key)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        message = payload.get("msg") or payload.get("code") or "non-list response"
        raise DataUnavailable(f"{endpoint} returned {message}")
    if not isinstance(payload, list):
        raise DataUnavailable(f"{endpoint} returned an unexpected response")
    return payload


def _fetch_endpoint(
    symbol: str,
    endpoint: str,
    start_ms: int,
    end_ms: int,
    api_key: str,
) -> tuple[dict[int, dict[str, Any]], str]:
    records: dict[int, dict[str, Any]] = {}
    cursor_end = end_ms - 1
    last_error: Exception | None = None

    while cursor_end >= start_ms:
        batch: list[dict[str, Any]] | None = None
        used_base: str | None = None
        for base_url in BASE_URLS:
            try:
                batch = _request(
                    base_url, endpoint, symbol, start_ms, cursor_end, api_key
                )
                used_base = base_url
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, DataUnavailable) as exc:
                last_error = exc
        if batch is None or used_base is None:
            raise DataUnavailable(
                f"could not retrieve {symbol} {endpoint}: {last_error}"
            )

        for row in batch:
            try:
                timestamp = int(row["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataUnavailable(f"invalid timestamp in {endpoint}") from exc
            if start_ms <= timestamp < end_ms:
                records[timestamp] = row

        if not batch:
            break
        oldest = min(int(row["timestamp"]) for row in batch if "timestamp" in row)
        if len(batch) < MAX_LIMIT or oldest <= start_ms:
            break
        next_cursor = oldest - 1
        if next_cursor >= cursor_end:
            raise DataUnavailable(f"pagination did not advance for {endpoint}")
        cursor_end = next_cursor

    return records, used_base


def _number(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            value = float(row[name])
            if math.isfinite(value):
                return value
    raise ValueError(f"missing numeric field: {names}")


def fetch_symbol(
    symbol: str,
    since: datetime,
    until: datetime,
    output_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_1h.csv"
    manifest_path = output_dir / f"{symbol}_1h.manifest.json"
    empty_rows: list[dict[str, str]] = []

    base_manifest: dict[str, Any] = {
        "source": "Binance USD-M futures market-data API",
        "symbol": symbol,
        "period": "1h",
        "requested_start": since.isoformat(),
        "requested_end_exclusive": until.isoformat(),
        "api_key_required": False,
        "rolling_history_limit_days": 30,
        "available": False,
        "status": "blocked",
        "row_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    start_ms = int(since.timestamp() * 1000)
    end_ms = int(until.timestamp() * 1000)
    try:
        account, account_base = _fetch_endpoint(
            symbol, ENDPOINTS["account"], start_ms, end_ms, api_key
        )
        position, position_base = _fetch_endpoint(
            symbol, ENDPOINTS["position"], start_ms, end_ms, api_key
        )
    except (DataUnavailable, ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        base_manifest["reason"] = str(exc)
        manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
        return base_manifest

    rows: list[dict[str, str]] = []
    for timestamp in sorted(set(account) & set(position)):
        try:
            account_row = account[timestamp]
            position_row = position[timestamp]
            account_long = _number(account_row, "longAccount")
            account_short = _number(account_row, "shortAccount")
            position_long = _number(
                position_row, "longAccount", "longPosition", "longPositionRatio"
            )
            position_short = _number(
                position_row, "shortAccount", "shortPosition", "shortPositionRatio"
            )
            account_ratio = _number(account_row, "longShortRatio")
            position_ratio = _number(position_row, "longShortRatio")
            values = (
                account_long,
                account_short,
                position_long,
                position_short,
                account_ratio,
                position_ratio,
            )
            if (
                not all(math.isfinite(value) for value in values)
                or not 0 <= account_long <= 1
                or not 0 <= account_short <= 1
                or not 0 <= position_long <= 1
                or not 0 <= position_short <= 1
                or account_ratio <= 0
                or position_ratio <= 0
            ):
                continue
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        timestamp / 1000, tz=timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    "account_long": str(account_long),
                    "account_short": str(account_short),
                    "position_long": str(position_long),
                    "position_short": str(position_short),
                    "account_long_short_ratio": str(account_ratio),
                    "position_long_short_ratio": str(position_ratio),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    base_manifest.update(
        {
            "available": bool(rows),
            "status": "available" if rows else "blocked",
            "reason": None if rows else "no aligned account and position observations",
            "row_count": len(rows),
            "first_timestamp": rows[0]["timestamp"] if rows else None,
            "last_timestamp": rows[-1]["timestamp"] if rows else None,
            "account_endpoint_base": account_base,
            "position_endpoint_base": position_base,
            "endpoint_sha256": hashlib.sha256(
                (ENDPOINTS["account"] + ENDPOINTS["position"]).encode()
            ).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    return base_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--since", required=True, help="inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="exclusive UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/historical/binance/top_trader"),
    )
    args = parser.parse_args()

    since = _day_start(args.since)
    until = _day_start(args.until)
    if since >= until:
        raise ValueError("--since must be earlier than --until")
    result = {
        symbol: fetch_symbol(
            symbol,
            since,
            until,
            args.output_dir,
            os.environ.get("BINANCE_API_KEY", "").strip(),
        )
        for symbol in args.symbols
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
