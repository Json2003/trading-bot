#!/usr/bin/env python3
"""Fetch reproducible daily cross-asset regime and Binance execution data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


BINANCE_BASE_URL = "https://data.binance.vision"
YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
MACRO_SYMBOLS = {"SPY": "SPY", "QQQ": "QQQ", "TLT": "TLT", "UUP": "UUP", "VIX": "^VIX"}
CRYPTO_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
NY = ZoneInfo("America/New_York")


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
    request = urllib.request.Request(url, headers={"User-Agent": "trading-bot-cross-asset-research/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _checksum(payload: bytes, checksum_text: str) -> str:
    expected = checksum_text.strip().split()[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"unrecognized checksum: {checksum_text!r}")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    return actual


def _parse_binance_archive(payload: bytes, symbol: str, start: date, end: date) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one Binance CSV for {symbol}, found {names}")
        rows: list[dict[str, str]] = []
        with archive.open(names[0], "r") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for values in reader:
                if not values or values[0].lower().replace(" ", "_") in {"open_time", "open_time_"}:
                    continue
                if len(values) < 6:
                    raise ValueError(f"short Binance kline row for {symbol}")
                timestamp_ms = int(float(values[0]))
                current_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()
                if not start <= current_date < end:
                    continue
                prices = [float(value) for value in values[1:5]]
                volume = float(values[5])
                if (
                    not all(math.isfinite(value) and value > 0 for value in prices)
                    or not math.isfinite(volume)
                    or volume < 0
                ):
                    raise ValueError(f"invalid Binance OHLCV row for {symbol} {current_date}")
                rows.append({
                    "date": current_date.isoformat(),
                    "open": values[1], "high": values[2], "low": values[3],
                    "close": values[4], "volume": values[5],
                })
    return rows


def fetch_binance(symbol: str, since: str, until: str, raw_dir: Path, output_dir: Path) -> dict[str, object]:
    start = datetime.strptime(since, "%Y-%m").date().replace(day=1)
    until_dt = datetime.strptime(until, "%Y-%m").date().replace(day=1)
    raw_symbol_dir = raw_dir / symbol
    raw_symbol_dir.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, dict[str, str]] = {}
    archives: list[dict[str, object]] = []
    for year, month in month_iter(since, until):
        stem = f"{symbol}-1d-{year:04d}-{month:02d}"
        url = f"{BINANCE_BASE_URL}/data/spot/monthly/klines/{symbol}/1d/{stem}.zip"
        try:
            payload = _download(url)
            checksum_text = _download(f"{url}.CHECKSUM").decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Binance archive unavailable for {symbol} {stem}: HTTP {exc.code}") from exc
        digest = _checksum(payload, checksum_text)
        month_rows = _parse_binance_archive(payload, symbol, start, until_dt)
        for row in month_rows:
            previous = by_date.get(row["date"])
            if previous is not None and previous != row:
                raise ValueError(f"conflicting Binance duplicate for {symbol} {row['date']}")
            by_date[row["date"]] = row
        archives.append({"url": url, "sha256": digest, "rows": len(month_rows), "month": stem})
    rows = [by_date[key] for key in sorted(by_date)]
    if not rows:
        raise ValueError(f"no Binance rows found for {symbol}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{symbol.replace('USDT', '')}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": BINANCE_BASE_URL, "symbol": symbol, "interval": "1d",
        "since": since, "until_exclusive": until, "archives": archives,
        "row_count": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
    }
    (output_dir / f"{symbol.replace('USDT', '')}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output": str(output), **{key: value for key, value in manifest.items() if key != "archives"}}


def fetch_yahoo(name: str, ticker: str, since: str, until: str, output_dir: Path) -> dict[str, object]:
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    query = urllib.parse.urlencode({
        "period1": int(start.timestamp()), "period2": int(end.timestamp()),
        "interval": "1d", "events": "history", "includeAdjustedClose": "true",
    })
    url = f"{YAHOO_BASE_URL}/{urllib.parse.quote(ticker, safe='') }?{query}"
    payload = json.loads(_download(url).decode("utf-8"))
    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo returned no chart for {ticker}: {error}")
    chart = result[0]
    timestamps = chart.get("timestamp", [])
    quote = chart.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    adjusted = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", quote)
    rows: list[dict[str, str]] = []
    for timestamp, close, adj_close in zip(timestamps, quote, adjusted):
        if close is None:
            continue
        value = float(close if ticker == "^VIX" or adj_close is None else adj_close)
        if not math.isfinite(value) or value <= 0:
            continue
        local_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(NY).date()
        rows.append({"date": local_date.isoformat(), "close": str(value)})
    unique = {row["date"]: row for row in rows}
    rows = [unique[key] for key in sorted(unique)]
    if len(rows) < 50:
        raise ValueError(f"insufficient Yahoo rows for {name}: {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{name}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "close"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": YAHOO_BASE_URL, "ticker": ticker, "since": since, "until_exclusive": until,
        "row_count": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "price_field": "close for VIX; adjusted close for ETFs",
    }
    (output_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2021-01")
    parser.add_argument("--until", default="2026-08", help="exclusive completed month")
    parser.add_argument("--output-dir", type=Path, default=Path("data/historical/cross_asset"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/historical/cross_asset/raw"))
    args = parser.parse_args()
    result: dict[str, object] = {}
    for name, ticker in MACRO_SYMBOLS.items():
        result[name] = fetch_yahoo(name, ticker, f"{args.since}-01", f"{args.until}-01", args.output_dir)
    for name, symbol in CRYPTO_SYMBOLS.items():
        result[name] = fetch_binance(symbol, args.since, args.until, args.raw_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
