"""Fetch klines from Binance REST API and upload to Google Cloud Storage.

Example usage:
  python binance_to_gcs.py --bucket mybucket --symbols BTCUSDT,ETHUSDT \
      --intervals 1m,5m --markets spot,um \
      --start 2024-01-01T00:00:00 --end 2024-01-01T01:00:00
"""
import argparse
import datetime as dt
import io
import logging
from typing import List

import pandas as pd
import requests
from google.cloud import storage

SPOT_URL = "https://api.binance.com/api/v3/klines"
UM_URL = "https://fapi.binance.com/fapi/v1/klines"

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
}


def _base_url(market: str) -> str:
    if market == "spot":
        return SPOT_URL
    if market in {"um", "futures"}:
        return UM_URL
    raise ValueError(f"Unsupported market {market}")


def fetch_klines(symbol: str, interval: str, start: int, end: int, market: str) -> List[list]:
    url = _base_url(market)
    out: List[list] = []
    cur = start
    step = INTERVAL_MS.get(interval)
    if step is None:
        raise ValueError(f"Unsupported interval {interval}")
    while cur < end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        out.extend(data)
        cur = data[-1][0] + step
    return out


def download_and_upload(bucket_name: str, symbols: List[str], intervals: List[str],
                        markets: List[str], start: dt.datetime, end: dt.datetime,
                        dest_path: str = "raw/binance"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    for market in markets:
        for symbol in symbols:
            for interval in intervals:
                logging.info("Fetching %s %s %s", market, symbol, interval)
                klines = fetch_klines(symbol, interval, start_ms, end_ms, market)
                if not klines:
                    logging.warning("No data for %s %s %s", market, symbol, interval)
                    continue
                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ]
                df = pd.DataFrame(klines, columns=cols)
                data_str = df.to_csv(index=False)
                blob_name = (
                    f"{dest_path}/{market}/{symbol}/{interval}/"
                    f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}.csv"
                )
                bucket.blob(blob_name).upload_from_string(data_str)
                logging.info("Uploaded %s", blob_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="GCS bucket name")
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--intervals", required=True,
                    help="Comma-separated intervals, e.g. 1m,5m")
    ap.add_argument("--markets", default="spot",
                    help="Comma-separated markets: spot or um (futures)")
    ap.add_argument("--start", required=True,
                    help="Start time YYYY-MM-DDTHH:MM:SS (UTC)")
    ap.add_argument("--end", required=True,
                    help="End time YYYY-MM-DDTHH:MM:SS (UTC)")
    ap.add_argument("--dest-path", default="raw/binance",
                    help="Destination path prefix in bucket")
    ap.add_argument("--log", default="INFO", help="Logging level")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    download_and_upload(args.bucket, symbols, intervals, markets, start, end, args.dest_path)


if __name__ == "__main__":
    main()
