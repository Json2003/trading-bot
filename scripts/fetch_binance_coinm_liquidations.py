#!/usr/bin/env python3
"""Download and normalize Binance COIN-M liquidation snapshots.

The source is Binance Vision's daily COIN-M liquidationSnapshot archive. The
archive coverage is intentionally bounded by the frozen experiment window;
missing daily archives are errors, never zero-liquidation observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://data.binance.vision"
CONTRACT_SIZE_USD = {"BTCUSD_PERP": 100.0, "ETHUSD_PERP": 10.0}
FIELDNAMES = ("timestamp", "side", "liquidation_usd")


def day_iter(since: str, until: str) -> list[datetime]:
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("--since must be earlier than exclusive --until")
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "trading-bot-liquidation-research/1.0"}
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
    value = int(float(raw))
    if value >= 10**14:
        value //= 1000
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _read_archive(zip_bytes: bytes, contract_size: float) -> list[dict[str, object]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in archive, found {names}")
        rows: list[dict[str, object]] = []
        with archive.open(names[0], "r") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            required = {
                "time",
                "side",
                "original_quantity",
                "last_fill_quantity",
                "accumulated_fill_quantity",
                "order_status",
            }
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(
                    f"liquidation CSV missing columns: {sorted(required - set(reader.fieldnames or []))}"
                )
            for row in reader:
                if row["order_status"].upper() != "FILLED":
                    continue
                side = row["side"].upper()
                if side not in {"BUY", "SELL"}:
                    raise ValueError(f"unexpected liquidation side: {side}")
                quantity = float(row["accumulated_fill_quantity"])
                if quantity <= 0:
                    raise ValueError("filled liquidation quantity must be positive")
                rows.append(
                    {
                        "timestamp": _timestamp(row["time"]).isoformat().replace("+00:00", "Z"),
                        "side": side,
                        "liquidation_usd": quantity * contract_size,
                    }
                )
        return rows


def fetch_symbol(
    symbol: str,
    since: str,
    until: str,
    output_dir: Path,
) -> dict[str, object]:
    if symbol not in CONTRACT_SIZE_USD:
        raise ValueError(f"unsupported COIN-M liquidation symbol: {symbol}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    archives: list[dict[str, object]] = []
    for day in day_iter(since, until):
        date = day.strftime("%Y-%m-%d")
        stem = f"{symbol}-liquidationSnapshot-{date}"
        url = (
            f"{BASE_URL}/data/futures/cm/daily/liquidationSnapshot/"
            f"{symbol}/{stem}.zip"
        )
        checksum_url = f"{url}.CHECKSUM"
        try:
            zip_bytes = _download(url)
            checksum_text = _download(checksum_url).decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"required liquidation archive unavailable for {symbol} {date}: HTTP {exc.code}"
            ) from exc
        digest = _checksum(zip_bytes, checksum_text)
        day_rows = _read_archive(zip_bytes, CONTRACT_SIZE_USD[symbol])
        rows.extend(day_rows)
        archives.append({"url": url, "sha256": digest, "rows": len(day_rows)})
    rows.sort(key=lambda row: (str(row["timestamp"]), str(row["side"])))
    path = output_dir / f"{symbol}_liquidations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": BASE_URL,
        "market": "COIN-M futures",
        "symbol": symbol,
        "contract_size_usd": CONTRACT_SIZE_USD[symbol],
        "since": since,
        "until_exclusive": until,
        "archive_count": len(archives),
        "row_count": len(rows),
        "archives": archives,
    }
    (output_dir / f"{symbol}_liquidations.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {"path": str(path), **{key: value for key, value in manifest.items() if key != "archives"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(CONTRACT_SIZE_USD))
    parser.add_argument("--since", required=True, help="inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="exclusive UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/historical/binance/liquidations"),
    )
    args = parser.parse_args()
    result = {
        symbol: fetch_symbol(symbol, args.since, args.until, args.output_dir)
        for symbol in args.symbols
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
