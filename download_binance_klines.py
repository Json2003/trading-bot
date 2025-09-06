#!/usr/bin/env python3
"""Download daily kline zip files from data.binance.vision.

This helper retrieves zipped candlestick data for a given symbol and interval
across a specified month.  By default it downloads BTCUSDT 1m data for
January 2024.
"""
import argparse
import os
import urllib.error
import urllib.request

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"

def download_day(symbol: str, interval: str, year: int, month: int, day: int,
                 out_dir: str, timeout: int) -> None:
    """Download a single day's zip file."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{year}/{month:02d}/{filename}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            data = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(e) from e
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "wb") as f:
        f.write(data)

def main() -> None:
    ap = argparse.ArgumentParser(description="Download Binance kline zip files")
    ap.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    ap.add_argument("--interval", default="1m", help="Kline interval (e.g. 1m, 5m)")
    ap.add_argument("--year", type=int, default=2024, help="Year of data")
    ap.add_argument("--month", type=int, default=1, help="Month of data (1-12)")
    ap.add_argument("--start-day", type=int, default=1, help="Starting day of month")
    ap.add_argument("--end-day", type=int, default=31, help="Ending day of month (inclusive)")
    ap.add_argument("--out-dir", default="data", help="Directory to save zip files")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP request timeout in seconds")
    args = ap.parse_args()

    for day in range(args.start_day, args.end_day + 1):
        try:
            download_day(args.symbol, args.interval, args.year, args.month, day,
                         args.out_dir, args.timeout)
        except Exception as e:  # pragma: no cover - logging for CLI usage
            print(f"Failed to download day {day:02d}: {e}")

if __name__ == "__main__":
    main()
