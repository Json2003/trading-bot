#!/usr/bin/env python3
"""Download and normalize official Binance Vision kline archives.

The downloader uses public archive files only; it never needs exchange keys.
Monthly archives are preferred and missing months fall back to daily archives.
Every output is normalized to UTC OHLCV CSV with a stable schema:
``timestamp,open,high,low,close,volume``.

Example (one year plus a warm-up month at hourly resolution)::

    python scripts/fetch_binance_klines.py \
      --symbols BTCUSDT ETHUSDT SOLUSDT --interval 1h \
      --start 2024-01-01 --end 2025-02-01 \
      --raw-root data/raw/binance-vision --output-root data/historical/binance

The source archive layout and checksum files are documented by Binance at
https://github.com/binance/binance-public-data.
"""

from __future__ import annotations

import argparse
from calendar import monthrange
from datetime import date, datetime, timedelta
import gzip
import hashlib
import io
import importlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _third_party(name: str):
    original = sys.path.copy()
    try:
        repo_paths = {path for path in original if str(REPO_ROOT) in os.path.abspath(path)}
        site_paths = [
            path
            for path in original
            if "site-packages" in (path or "") or "dist-packages" in (path or "")
        ]
        remaining = [path for path in original if path not in repo_paths and path not in site_paths]
        sys.path[:] = site_paths + remaining
        if name in sys.modules:
            del sys.modules[name]
        return importlib.import_module(name)
    finally:
        sys.path[:] = original


pd = _third_party("pandas")
try:
    requests = _third_party("requests")
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ModuleNotFoundError:  # pragma: no cover - exercised only in the offline shim image
    requests = None
    HTTPAdapter = None
    Retry = None


BASE_URL = "https://data.binance.vision"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
MARKET_PATHS = {"spot": "spot", "futures_um": "futures/um"}


class ArchiveUnavailable(RuntimeError):
    """Raised when Binance has no archive at a requested URL."""


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def normalize_epoch_unit(values: pd.Series) -> str:
    """Infer seconds, milliseconds, or microseconds without date assumptions."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("kline timestamp column is empty")
    sample = abs(float(numeric.iloc[0]))
    if sample >= 1e15:
        return "us"
    if sample >= 1e12:
        return "ms"
    if sample >= 1e9:
        return "s"
    raise ValueError(f"unsupported epoch timestamp magnitude: {sample}")


def archive_url(
    market: str,
    symbol: str,
    interval: str,
    when: date,
    *,
    monthly: bool,
) -> str:
    """Build an official Binance Vision monthly or daily kline URL."""

    try:
        market_path = MARKET_PATHS[market]
    except KeyError as exc:
        raise ValueError(f"unsupported market: {market}") from exc
    symbol = symbol.upper()
    if monthly:
        filename = f"{symbol}-{interval}-{when:%Y-%m}.zip"
        return f"{BASE_URL}/data/{market_path}/monthly/klines/{symbol}/{interval}/{filename}"
    filename = f"{symbol}-{interval}-{when:%Y-%m-%d}.zip"
    return (
        f"{BASE_URL}/data/{market_path}/daily/klines/{symbol}/{interval}/"
        f"{when:%Y}/{when:%m}/{when:%d}/{filename}"
    )


def iter_months(start: date, end: date):
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        yield current
        current = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def make_session() -> requests.Session:
    if requests is None or HTTPAdapter is None or Retry is None:
        raise RuntimeError("the requests and urllib3 packages are required for downloads")
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "trading-bot-binance-vision-klines/1.0"})
    return session


def _download(url: str, destination: Path, session: requests.Session) -> bool:
    """Download one archive atomically and return whether it was new."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return False
    partial = destination.with_name(destination.name + ".part")
    try:
        with session.get(url, stream=True, timeout=(15, 120)) as response:
            if response.status_code == 404:
                raise ArchiveUnavailable(url)
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError(f"empty archive response: {url}")
        partial.replace(destination)
        return True
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _checksum(url: str, archive: Path, session: requests.Session) -> str:
    response = session.get(url + ".CHECKSUM", timeout=(15, 30))
    if response.status_code == 404:
        raise ArchiveUnavailable(url + ".CHECKSUM")
    response.raise_for_status()
    expected_match = re.search(r"([0-9a-fA-F]{64})", response.text)
    if not expected_match:
        raise ValueError(f"checksum response did not contain SHA256: {url}.CHECKSUM")
    expected = expected_match.group(1).lower()
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {archive.name}: {actual} != {expected}")
    return actual


def _read_member(payload: bytes, name: str) -> pd.DataFrame:
    raw = gzip.decompress(payload) if name.endswith(".gz") else payload
    sample = raw[:512].decode("utf-8", errors="replace")
    has_header = bool(sample.splitlines() and not sample.splitlines()[0].split(",", 1)[0].strip().isdigit())
    frame = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if not has_header:
        frame.columns = KLINE_COLUMNS[: len(frame.columns)]
    else:
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        aliases = {"quote_asset_volume": "quote_volume", "taker_buy_base_asset_volume": "taker_buy_base_volume", "taker_buy_quote_asset_volume": "taker_buy_quote_volume"}
        frame = frame.rename(columns=aliases)
    missing = set(KLINE_COLUMNS[:6]) - set(frame.columns)
    if missing:
        raise ValueError(f"kline archive member {name} is missing columns: {sorted(missing)}")
    frame = frame[KLINE_COLUMNS[:6]].copy()
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["open_time"], errors="coerce"),
        unit=normalize_epoch_unit(frame["open_time"]),
        utc=True,
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=OUTPUT_COLUMNS)
    return frame[OUTPUT_COLUMNS]


def read_kline_archive(path: Path) -> pd.DataFrame:
    """Read all CSV members from one downloaded Vision ZIP archive."""

    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".csv", ".csv.gz"))
        ]
        if not members:
            raise ValueError(f"no CSV kline member found in {path}")
        for member in members:
            frames.append(_read_member(archive.read(member), member.lower()))
    return pd.concat(frames, ignore_index=True)


def normalize_archives(
    archives: list[Path],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Merge archives, normalize timestamps, and clip to an inclusive date range."""

    if not archives:
        raise ValueError("no downloaded archives")
    frame = pd.concat([read_kline_archive(path) for path in archives], ignore_index=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_exclusive = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    frame = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_exclusive)]
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("downloaded archives contain no rows in the requested date range")
    return frame[OUTPUT_COLUMNS]


def fetch_symbol(
    *,
    market: str,
    symbol: str,
    interval: str,
    start: date,
    end: date,
    raw_root: Path,
    session: requests.Session,
    verify_checksums: bool,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Fetch monthly archives, falling back to daily files when necessary."""

    symbol_root = raw_root / market / symbol.upper() / interval
    archives: list[Path] = []
    manifest: list[dict[str, str]] = []
    for month in iter_months(start, end):
        url = archive_url(market, symbol, interval, month, monthly=True)
        destination = symbol_root / Path(url).name
        try:
            _download(url, destination, session)
            checksum = _checksum(url, destination, session) if verify_checksums else "skipped"
            archives.append(destination)
            manifest.append({"url": url, "path": str(destination), "sha256": checksum})
            continue
        except ArchiveUnavailable:
            destination.unlink(missing_ok=True)

        month_start = max(start, month)
        month_end = min(end, date(month.year, month.month, monthrange(month.year, month.month)[1]))
        for day in iter_days(month_start, month_end):
            daily_url = archive_url(market, symbol, interval, day, monthly=False)
            daily_destination = symbol_root / Path(daily_url).name
            try:
                _download(daily_url, daily_destination, session)
                checksum = (
                    _checksum(daily_url, daily_destination, session) if verify_checksums else "skipped"
                )
            except ArchiveUnavailable:
                daily_destination.unlink(missing_ok=True)
                continue
            archives.append(daily_destination)
            manifest.append({"url": daily_url, "path": str(daily_destination), "sha256": checksum})
    if not archives:
        raise RuntimeError(f"no Binance Vision archives found for {symbol} {interval}")
    return archives, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=sorted(MARKET_PATHS), default="spot")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/binance-vision"))
    parser.add_argument("--output-root", type=Path, default=Path("data/historical/binance"))
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="skip official .CHECKSUM verification (not recommended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")
    session = make_session()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_manifest: dict[str, object] = {
        "market": args.market,
        "interval": args.interval,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "checksum_verification": not args.no_checksum,
        "symbols": {},
    }
    for symbol in args.symbols:
        normalized_symbol = symbol.upper()
        archives, manifest = fetch_symbol(
            market=args.market,
            symbol=normalized_symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            raw_root=args.raw_root,
            session=session,
            verify_checksums=not args.no_checksum,
        )
        frame = normalize_archives(archives, start=args.start, end=args.end)
        output = args.output_root / normalized_symbol / f"{args.interval}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
        all_manifest["symbols"][normalized_symbol] = {
            "rows": len(frame),
            "output": str(output),
            "archives": manifest,
            "first_timestamp": frame["timestamp"].iloc[0].isoformat(),
            "last_timestamp": frame["timestamp"].iloc[-1].isoformat(),
        }
        print(f"{normalized_symbol}: {len(frame):,} rows -> {output}")
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(all_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
