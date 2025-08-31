"""Download data from Binance Vision and upload to Google Cloud Storage.

Example usage:
  python binance_vision_to_gcs.py --bucket mybucket --symbol BTCUSDT \
      --start 2024-01-01 --end 2024-01-03 --interval 1m
"""
import argparse
import datetime as dt
import io
import logging
import zipfile

import requests
from google.cloud import storage

BASE_URL = "https://data.binance.vision/"


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def download_and_upload(bucket_name: str, symbol: str, start: dt.date, end: dt.date,
                         interval: str = "1m", dest_path: str = "raw/binance"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for day in daterange(start, end):
        fname = f"{symbol}-{interval}-{day.strftime('%Y-%m-%d')}.zip"
        url = f"{BASE_URL}data/spot/daily/klines/{symbol}/{interval}/{fname}"
        logging.info("Downloading %s", url)
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            logging.warning("Missing %s", url)
            continue
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                data = zf.read(member)
                blob_name = f"{dest_path}/{symbol}/{member}"
                bucket.blob(blob_name).upload_from_string(data)
                logging.info("Uploaded %s", blob_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="GCS bucket name")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--interval", default="1m", help="Kline interval (default 1m)")
    ap.add_argument("--dest-path", default="raw/binance", help="Destination path prefix in bucket")
    ap.add_argument("--log", default="INFO", help="Logging level")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    download_and_upload(args.bucket, args.symbol, start, end, args.interval, args.dest_path)


if __name__ == "__main__":
    main()
