#!/usr/bin/env python3
"""
Convert Binance Vision raw CSV.GZ klines into partitioned Parquet files.

Partitions layout:
  <out_root>/symbol=<SYMBOL>/date=YYYY-MM-DD/part.parquet

Defaults (can be overridden by env vars or flags):
  BINANCE_RAW_DIR = data/raw/binance/spot
  PARQUET_OUT     = data/parquet/ohlcv_1m
  SYMBOLS         = BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT

Examples:
  python scripts/binance_raw_to_parquet.py \
    --raw data/raw/binance/um/futures \
    --out data/parquet/um/ohlcv_1m \
    --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations
import os
import gzip
import glob
import argparse
from pathlib import Path
import logging
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def import_third_party(mod_name: str):
    import importlib
    original = sys.path.copy()
    try:
        repo_paths = {p for p in original if str(REPO_ROOT) in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        return importlib.import_module(mod_name)
    finally:
        sys.path = original


pd = import_third_party("pandas")

logger = logging.getLogger("binance_parquet")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def read_csv_gz(path: str):
    """Read a gzip CSV from Binance Vision (kline) to a normalized DataFrame.

    Columns expected (no header):
    open_time, open, high, low, close, volume, close_time, qav, num_trades, taker_base, taker_quote, ignore
    """
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(
            f,
            header=None,
            names=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "num_trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"]) 
    df = df.astype({
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    })
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    return df


def write_partitions(symbol: str, df, out_root: Path):
    df = df.copy()
    df["date"] = df["ts"].dt.date.astype("string")
    for date, g in df.groupby("date", sort=True):
        out = out_root / f"symbol={symbol}" / f"date={date}"
        out.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["date"]).to_parquet(out / "part.parquet", index=False)


def process_symbol(symbol: str, raw_root: Path, out_root: Path):
    pattern = os.path.join(str(raw_root), symbol, "**", "*.csv.gz")
    files = glob.glob(pattern, recursive=True)
    if not files:
        logger.warning(f"No files found for %s under %s", symbol, raw_root)
        return
    parts = []
    for fp in sorted(files):
        parts.append(read_csv_gz(fp))
    df = pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates("ts")
    write_partitions(symbol, df, out_root)
    logger.info("Wrote Parquet for %s (rows=%d)", symbol, len(df))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Binance raw CSV.GZ klines to partitioned Parquet")
    p.add_argument("--raw", default=os.environ.get("BINANCE_RAW_DIR", "data/raw/binance/spot"), help="Input raw directory root")
    p.add_argument("--out", default=os.environ.get("PARQUET_OUT", "data/parquet/ohlcv_1m"), help="Output Parquet root directory")
    p.add_argument("--symbols", nargs="*", default=os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT").split(","), help="Symbols to process")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_root = Path(args.raw)
    out_root = Path(args.out)
    symbols = [s.strip() for s in args.symbols if s and s.strip()]

    out_root.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        process_symbol(s, raw_root, out_root)


if __name__ == "__main__":
    main()
