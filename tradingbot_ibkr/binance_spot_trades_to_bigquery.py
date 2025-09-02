"""Download Binance spot trades, upload to GCS as Parquet, and register a
BigQuery external table.

The script discovers USDT trading pairs listed in the Binance Vision daily
trade index.  For each symbol and month in the requested date range it fetches
compressed CSV trade dumps, converts them to a normalized Parquet file, and
uploads the file to the specified Google Cloud Storage bucket.  After data is
uploaded an external BigQuery table pointing at the bucket is created (or
replaced).

Example:
  python binance_spot_trades_to_bigquery.py \
      --bucket my-bucket \
      --project my-project \
      --dataset binance \
      --table spot_trades \
      --region us-central1
"""
from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from datetime import datetime
from typing import Iterable

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, storage
from google.cloud import bigquery_connection_v1

BASE = "https://data.binance.vision"
DAILY_INDEX = "/data/spot/daily/trades/"
MONTHLY_TMPL = "/data/spot/monthly/trades/{symbol}/{symbol}-trades-{yyyy}-{mm}.zip"
DAILY_TMPL = "/data/spot/daily/trades/{symbol}/{yyyy}/{mm}/{symbol}-trades-{yyyy}-{mm}-{dd}.zip"


def month_iter(since: str, until: str) -> Iterable[tuple[str, str]]:
    cur = datetime.strptime(since, "%Y-%m")
    end = datetime.strptime(until, "%Y-%m")
    while cur <= end:
        yield cur.strftime("%Y"), cur.strftime("%m")
        cur += relativedelta(months=1)


def month_days(yyyy: str, mm: str) -> Iterable[str]:
    d0 = datetime.strptime(f"{yyyy}-{mm}-01", "%Y-%m-%d")
    d1 = d0 + relativedelta(months=1)
    for d in range(1, (d1 - d0).days + 1):
        yield f"{d:02d}"


def http_head(url: str) -> bool:
    try:
        r = requests.head(url, timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def list_symbols_usdt() -> list[str]:
    url = BASE + DAILY_INDEX
    html = requests.get(url, timeout=60).text
    syms = set(re.findall(r'href="/data/spot/daily/trades/([A-Z0-9]+)/"', html))
    return sorted([s for s in syms if s.endswith("USDT")])


def upload_parquet(df: pd.DataFrame, bucket: storage.Bucket, symbol: str, year: str, month: str) -> None:
    df = df.copy()
    df["symbol"] = symbol
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    df.dropna(subset=["time"], inplace=True)
    for col in ("price", "qty"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["price", "qty"], inplace=True)
    cols = [c for c in ["time", "price", "qty", "quoteQty", "isBuyerMaker", "isBestMatch", "tradeId", "symbol"] if c in df.columns]
    if not cols:
        return
    df = df[cols]

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    path = f"binance/spot/trades/{symbol}/year={year}/month={month}/part-{symbol}-{year}-{month}.parquet"
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    bucket.blob(path).upload_from_file(buf, content_type="application/octet-stream")


def process_symbol(bucket: storage.Bucket, symbol: str, since: str, until: str) -> None:
    sess = requests.Session()
    for yyyy, mm in month_iter(since, until):
        murl = BASE + MONTHLY_TMPL.format(symbol=symbol, yyyy=yyyy, mm=mm)
        if http_head(murl):
            z = sess.get(murl, timeout=120).content
            with zipfile.ZipFile(io.BytesIO(z)) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        df = pd.read_csv(f)
                        upload_parquet(df, bucket, symbol, yyyy, mm)
            continue
        for dd in month_days(yyyy, mm):
            durl = BASE + DAILY_TMPL.format(symbol=symbol, yyyy=yyyy, mm=mm, dd=dd)
            if not http_head(durl):
                continue
            z = sess.get(durl, timeout=120).content
            with zipfile.ZipFile(io.BytesIO(z)) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        df = pd.read_csv(f)
                        upload_parquet(df, bucket, symbol, yyyy, mm)


def create_external_table(project: str, dataset: str, table: str, bucket: str, region: str, connection: str) -> None:
    bq_client = bigquery.Client(project=project)
    dataset_id = f"{project}.{dataset}"
    ds = bigquery.Dataset(dataset_id)
    ds.location = region
    bq_client.create_dataset(ds, exists_ok=True)

    table_id = f"{dataset_id}.{table}"
    external_config = bigquery.ExternalConfig("PARQUET")
    external_config.source_uris = [f"gs://{bucket}/binance/spot/trades/*/year=*/month=*/*.parquet"]
    external_config.connection_id = f"{region}.{connection}"
    table_obj = bigquery.Table(table_id)
    table_obj.external_data_configuration = external_config
    bq_client.create_table(table_obj, exists_ok=True)


def ensure_gcs_connection(project: str, region: str, connection_id: str) -> None:
    """Create a BigQuery Cloud Resource connection if it does not already exist."""
    client = bigquery_connection_v1.ConnectionServiceClient()
    parent = f"projects/{project}/locations/{region}"
    name = f"{parent}/connections/{connection_id}"
    try:
        client.get_connection(name=name)
    except NotFound:
        conn = bigquery_connection_v1.Connection(
            cloud_resource=bigquery_connection_v1.Connection.CloudResourceProperties()
        )
        client.create_connection(parent=parent, connection_id=connection_id, connection=conn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket to upload to")
    parser.add_argument("--project", required=True, help="GCP project for BigQuery")
    parser.add_argument("--dataset", default="binance", help="BigQuery dataset name")
    parser.add_argument("--table", default="spot_trades", help="BigQuery table name")
    parser.add_argument("--region", default="us", help="Location for BigQuery resources")
    parser.add_argument("--connection", default="gcs_conn", help="BigQuery connection id")
    parser.add_argument("--since", default="2017-01", help="Start month YYYY-MM")
    parser.add_argument("--until", default=datetime.utcnow().strftime("%Y-%m"), help="End month YYYY-MM")
    parser.add_argument("--symbols-regex", default=".*USDT$", help="Regex to filter symbols")
    args = parser.parse_args()

    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket)

    ensure_gcs_connection(args.project, args.region, args.connection)

    rx = re.compile(args.symbols_regex)
    symbols = [s for s in list_symbols_usdt() if rx.match(s)]
    for sym in symbols:
        process_symbol(bucket, sym, args.since, args.until)

    create_external_table(args.project, args.dataset, args.table, args.bucket, args.region, args.connection)


if __name__ == "__main__":
    main()
