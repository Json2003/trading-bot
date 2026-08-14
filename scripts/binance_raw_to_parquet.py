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
import io

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


def _epoch_unit(values) -> str:
    """Infer Binance seconds/milliseconds/microseconds timestamps."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("timestamp column is empty")
    sample = abs(float(numeric.iloc[0]))
    if sample >= 1e15:
        return "us"
    if sample >= 1e12:
        return "ms"
    if sample >= 1e9:
        return "s"
    raise ValueError(f"unsupported timestamp magnitude: {sample}")


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
    df["ts"] = pd.to_datetime(df["open_time"], unit=_epoch_unit(df["open_time"]), utc=True)
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


def read_trade_csv(path: str, limit_rows: int | None = None):
    """Read a trade-level CSV (possibly gzipped) and return a small DataFrame with ts, price, qty.

    The function will try common timestamp/price/qty column names used by Binance trade dumps.
    """
    compression = 'gzip' if str(path).lower().endswith('.gz') else None
    # pandas can handle gz if compression='gzip'
    if limit_rows is not None:
        df = pd.read_csv(path, compression=compression, nrows=limit_rows)
    else:
        df = pd.read_csv(path, compression=compression)

    # try to locate columns
    cols = {c.lower(): c for c in df.columns}
    ts_col = None
    for k in ('tradetime', 'tradetime_ms', 'tradetime_ms', 'tradetime_ms', 't', 'tradeTime', 'time', 'T', 't'):
        if k.lower() in cols:
            ts_col = cols[k.lower()]
            break
    if ts_col is None:
        # pick first numeric column as timestamp fallback
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                ts_col = c
                break

    price_col = None
    for k in ('price', 'p'):
        if k in cols:
            price_col = cols[k]
            break
    qty_col = None
    for k in ('qty', 'q', 'quantity', 'amount'):
        if k in cols:
            qty_col = cols[k]
            break

    if ts_col is None or price_col is None or qty_col is None:
        raise ValueError(f"Could not auto-detect ts/price/qty columns in {path}")

    df = df[[ts_col, price_col, qty_col]].rename(columns={ts_col: 'ts', price_col: 'price', qty_col: 'qty'})
    # Binance spot archives switched to microsecond timestamps for newer data.
    unit = _epoch_unit(df['ts'])
    df['ts'] = pd.to_datetime(df['ts'], unit=unit, utc=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df = df.dropna(subset=['ts', 'price', 'qty'])
    return df


def write_partitions(symbol: str, df, out_root: Path):
    df = df.copy()
    df["date"] = df["ts"].dt.date.astype("string")
    for date, g in df.groupby("date", sort=True):
        out = out_root / f"symbol={symbol}" / f"date={date}"
        out.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["date"]).to_parquet(out / "part.parquet", index=False)


def process_symbol(symbol: str, raw_root: Path, out_root: Path):
    # find klines and trade files
    kline_pattern = os.path.join(str(raw_root), symbol, "**", "*.csv.gz")
    trade_pattern = os.path.join(str(raw_root), symbol, "**", "*.csv")
    files = glob.glob(kline_pattern, recursive=True) or glob.glob(trade_pattern, recursive=True)
    if not files:
        logger.warning(f"No files found for %s under %s", symbol, raw_root)
        return
    parts = []
    for fp in sorted(files):
        fp = str(fp)
        try:
            if fp.lower().endswith('.csv'):
                # trade-level CSV
                parts.append(read_trade_csv(fp))
            else:
                parts.append(read_csv_gz(fp))
        except Exception as e:
            logger.warning('Skipping file %s: %s', fp, e)
    if not parts:
        logger.warning('No readable parts for %s', symbol)
        return
    df = pd.concat(parts, ignore_index=True)
    if 'price' in df.columns and 'qty' in df.columns:
        # trade-level data -> ensure sorted
        df = df.sort_values('ts').drop_duplicates(subset=['ts','price','qty'])
    else:
        df = df.sort_values('ts').drop_duplicates('ts')
    write_partitions(symbol, df, out_root)
    logger.info("Wrote Parquet for %s (rows=%d)", symbol, len(df))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Binance raw CSV.GZ klines to partitioned Parquet")
    p.add_argument("--raw", default=os.environ.get("BINANCE_RAW_DIR", "data/raw/binance/spot"), help="Input raw directory root")
    p.add_argument("--out", default=os.environ.get("PARQUET_OUT", "data/parquet/ohlcv_1m"), help="Output Parquet root directory")
    p.add_argument("--symbols", nargs="*", default=os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT").split(","), help="Symbols to process")
    p.add_argument('--aggregate-trades', action='store_true', help='If set, treat CSV files as trade dumps and aggregate to OHLCV before writing parquet')
    p.add_argument('--timeframe', default=os.environ.get('TIMEFRAME','1m'), help='Resampling timeframe when aggregating trades (e.g. 1m, 1h)')
    p.add_argument('--limit-rows', type=int, default=None, help='If set, limit rows read from each trade CSV (useful for quick demos)')
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_root = Path(args.raw)
    out_root = Path(args.out)
    symbols = [s.strip() for s in args.symbols if s and s.strip()]
    aggregate_trades = args.aggregate_trades
    timeframe = args.timeframe
    limit_rows = args.limit_rows

    out_root.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        if aggregate_trades:
            # process trade CSVs and aggregate
            pattern = os.path.join(str(raw_root), s, "**", "*.csv")
            files = glob.glob(pattern, recursive=True)
            if not files:
                logger.warning('No trade CSVs found for %s under %s', s, raw_root)
                continue
            parts = []
            for fp in sorted(files):
                try:
                    parts.append(read_trade_csv(fp, limit_rows))
                except Exception as e:
                    logger.warning('Skipping trade file %s: %s', fp, e)
            if not parts:
                logger.warning('No trade data parsed for %s', s)
                continue
            trades = pd.concat(parts, ignore_index=True)
            trades.set_index('ts', inplace=True)
            # resample to OHLCV
            ohlc = trades['price'].resample(timeframe).ohlc()
            vol = trades['qty'].resample(timeframe).sum()
            df = ohlc.join(vol.rename('volume')).reset_index().rename(columns={'index':'ts'})
            write_partitions(s, df, out_root)
            logger.info('Wrote aggregated Parquet for %s (rows=%d)', s, len(df))
        else:
            process_symbol(s, raw_root, out_root)


if __name__ == "__main__":
    main()
