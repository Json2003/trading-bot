#!/usr/bin/env python3
"""Ingest Binance trade dump files (tick-level) and optionally aggregate to OHLCV.

Usage examples (PowerShell):
  # parse all CSV/JSON files in a directory, save ticks and 1m OHLCV
  python binance_trade_dump_ingest.py --input-dir ./downloads --symbol BTC/USDT --out-dir ./datafiles --to-ohlcv 1m

  # just normalize ticks and append
  python binance_trade_dump_ingest.py --input-dir ./downloads --symbol BTC/USDT

The script supports CSV or JSON-lines where each record is a trade with at least
timestamp, price, and quantity fields. It will attempt to auto-detect common
Binance field names (e.g., 'tradeTime','time','T' for timestamp; 'price','p' for
price; 'qty','q','quantity' for quantity).
"""
import argparse
from pathlib import Path
import pandas as pd
import json
import sys
from typing import List, Optional
import hashlib
import concurrent.futures
import threading
import time
import os
import logging

# optional tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def find_files(input_dir: Path, pattern: str = "*") -> List[Path]:
    return sorted([p for p in input_dir.rglob(pattern) if p.is_file()])


def _file_id(path: Path) -> str:
    # small, stable id for processed-file tracking
    h = hashlib.sha1()
    h.update(str(path.name).encode('utf-8'))
    h.update(str(path.stat().st_size).encode('utf-8'))
    return h.hexdigest()


def read_trade_file(path: Path) -> pd.DataFrame:
    """Read a single trade dump (CSV or JSON lines) and return normalized DataFrame.

    Normalized columns: ts (datetime UTC), price (float), qty (float), side (str: 'buy'|'sell' or None)
    """
    text = path.suffix.lower()
    if text == '.csv':
        df = pd.read_csv(path)
    else:
        # try json lines
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        # try to parse as a JSON array first
        if lines and lines[0].strip().startswith('['):
            data = json.loads('\n'.join(lines))
            df = pd.DataFrame(data)
        else:
            # parse line-by-line JSON
            rows = [json.loads(l) for l in lines if l.strip()]
            df = pd.DataFrame(rows)

    # Normalize column names to lowercase for matching
    cols = {c: c.lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # detect timestamp column
    ts_candidates = ['tradeTime', 'tradetime', 'time', 't', 'T', 'timestamp', 'trade_time']
    ts_col = None
    for c in df.columns:
        if c.lower() in [x.lower() for x in ts_candidates]:
            ts_col = c
            break
    # detect price and qty
    price_candidates = ['price', 'p']
    qty_candidates = ['qty', 'q', 'quantity', 'amount']
    price_col = next((c for c in df.columns if c.lower() in price_candidates), None)
    qty_col = next((c for c in df.columns if c.lower() in qty_candidates), None)

    if ts_col is None:
        # try common names
        for c in df.columns:
            if 'time' in c.lower():
                ts_col = c
                break
    if price_col is None:
        for c in df.columns:
            if 'price' in c.lower():
                price_col = c
                break
    if qty_col is None:
        for c in df.columns:
            if 'qty' in c.lower() or 'quantity' in c.lower() or 'amount' in c.lower():
                qty_col = c
                break

    if ts_col is None or price_col is None or qty_col is None:
        raise ValueError(f"Could not detect required columns in {path}: ts_col={ts_col}, price_col={price_col}, qty_col={qty_col}")

    # handle timestamp formats: milliseconds integer or ISO
    s = df[ts_col]
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        # if values are large, assume ms
        example = int(s.dropna().iloc[0]) if len(s.dropna()) else 0
        if example > 1e10:
            df['ts'] = pd.to_datetime(s, unit='ms', utc=True)
        else:
            df['ts'] = pd.to_datetime(s, unit='s', utc=True)
    else:
        df['ts'] = pd.to_datetime(s, utc=True, errors='coerce')

    df['price'] = pd.to_numeric(df[price_col], errors='coerce')
    df['qty'] = pd.to_numeric(df[qty_col], errors='coerce')

    # side detection (buyer maker / isBuyerMaker / is_buyer_maker)
    side = None
    for cand in ['isBuyermaker', 'isBuyermaker'.lower(), 'is_buyermaker', 'isBuyermaker'.upper(), 'side']:
        if cand in df.columns:
            side = cand
            break
    if side:
        # normalize to 'buy'/'sell'
        df['side'] = df[side].map(lambda v: 'buy' if str(v).lower() in ['false','0','f','buyer'] or v is False else 'sell')
    else:
        df['side'] = None

    out = df[['ts', 'price', 'qty', 'side']].copy()
    out.dropna(subset=['ts','price','qty'], inplace=True)
    out.sort_values('ts', inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def append_ticks(ticks: pd.DataFrame, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=['ts'])
        existing['ts'] = pd.to_datetime(existing['ts'], utc=True)
        combined = pd.concat([existing, ticks])
    else:
        combined = ticks
    # dedupe
    combined.drop_duplicates(subset=['ts','price','qty'], inplace=True)
    combined.sort_values('ts', inplace=True)
    combined.to_csv(out_path, index=False)
    return len(ticks)


def ticks_to_ohlcv(ticks: pd.DataFrame, timeframe: str = '1m') -> pd.DataFrame:
    # map timeframe like '1m' -> '1min'
    tf_map = {'m':'min','h':'h','d':'D'}
    if timeframe[-1] in tf_map:
        unit = tf_map[timeframe[-1]]
        num = timeframe[:-1]
        pd_tf = f"{num}{unit}"
    else:
        pd_tf = timeframe

    df = ticks.set_index('ts').copy()
    df.index = pd.to_datetime(df.index, utc=True)
    o = df['price'].resample(pd_tf).ohlc()
    v = df['qty'].resample(pd_tf).sum()
    o['volume'] = v
    o.dropna(subset=['open','high','low','close'], inplace=True)
    o.reset_index(inplace=True)
    return o


def main():
    parser = argparse.ArgumentParser(description='Ingest Binance trade dump files (tick-level) and optionally aggregate to OHLCV')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--pattern', default='*')
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--out-dir', default='datafiles')
    parser.add_argument('--to-ohlcv', default=None, help="timeframe to resample ticks to (e.g. 1m, 1h)")
    parser.add_argument('--limit-files', type=int, default=None)
    parser.add_argument('--state-file', default='.ingest_state.json', help='path to state file to store processed file ids')
    parser.add_argument('--workers', type=int, default=None, help='number of worker threads to use (overrides INGEST_WORKERS env var)')
    parser.add_argument('--force', action='store_true', help='reprocess files even if present in state file')
    parser.add_argument('--log-file', default=None, help='path to logfile (if omitted, logs to stdout)')
    parser.add_argument('--progress', action='store_true', help='show progress bar if tqdm is installed')
    args = parser.parse_args()

    inp = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    files = find_files(inp, args.pattern)
    if args.limit_files:
        files = files[:args.limit_files]
    if not files:
        print('No files found in', inp)
        sys.exit(0)
    # compute size estimate for progress
    total_bytes = sum([p.stat().st_size for p in files])
    processed_bytes = 0
    processed_lock = threading.Lock()
    # determine worker count: CLI flag > env var > default 4
    if args.workers is not None:
        max_workers = max(1, int(args.workers))
    else:
        try:
            max_workers = int(os.getenv('INGEST_WORKERS', '4'))
        except Exception:
            max_workers = 4
    # if user passed via env var INGEST_WORKERS it will be used; otherwise default 4

    state_path = inp / args.state_file
    processed = set()
    if state_path.exists():
        try:
            processed = set(json.loads(state_path.read_text()))
        except Exception:
            processed = set()

    # configure logging
    logger = logging.getLogger('binance_ingest')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    all_ticks = []
    processed_ids = []

    def _worker(path: Path):
        nonlocal processed_bytes
        fid = _file_id(path)
        if fid in processed and not args.force:
            logger.info(f'Skipping already processed file {path.name}')
            with processed_lock:
                processed_bytes += path.stat().st_size
            return None, None
        try:
            t = read_trade_file(path)
            with processed_lock:
                processed_bytes += path.stat().st_size
                pct = (processed_bytes / total_bytes) * 100 if total_bytes else 100.0
            logger.info(f'Read {len(t)} ticks from {path.name}  [{pct:.1f}%]')
            # update tqdm if enabled
            if args.progress and tqdm is not None and 'bar' in globals() and bar is not None:
                try:
                    bar.update(path.stat().st_size)
                except Exception:
                    pass
            return fid, t
        except Exception as e:
            with processed_lock:
                processed_bytes += path.stat().st_size
            logger.warning(f'Failed to read {path}: {e}')
            return None, None

    # parse files concurrently
    bar = None
    try:
        if args.progress and tqdm is not None:
            bar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc='ingest')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_worker, p): p for p in files}
            for fut in concurrent.futures.as_completed(futures):
                fid, t = fut.result()
                if fid and t is not None:
                    all_ticks.append(t)
                    processed_ids.append(fid)
    finally:
        if bar is not None:
            try:
                bar.close()
            except Exception:
                pass

    if not all_ticks:
        print('No ticks parsed; exiting')
        sys.exit(0)

    ticks = pd.concat(all_ticks, ignore_index=True)
    ticks.drop_duplicates(subset=['ts','price','qty'], inplace=True)
    ticks.sort_values('ts', inplace=True)

    # write ticks
    out_ticks = out_dir / f"{args.symbol.replace('/','_')}_trades.csv"
    appended = append_ticks(ticks, out_ticks)
    logger.info(f'Appended {appended} ticks to {out_ticks}')

    # update state file
    processed.update(processed_ids)
    try:
        state_path.write_text(json.dumps(list(processed)))
    except Exception as e:
        logger.warning('Could not write state file: %s', e)

    # optionally resample to OHLCV and append to bars file
    if args.to_ohlcv:
        ohlcv = ticks_to_ohlcv(ticks, args.to_ohlcv)
        out_bars = out_dir / f"{args.symbol.replace('/','_')}_bars.csv"
        # merge with existing bars if present
        if out_bars.exists():
            existing = pd.read_csv(out_bars, parse_dates=['ts'], index_col='ts')
            existing.index = pd.to_datetime(existing.index, utc=True)
            new = ohlcv.set_index('ts')
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep='first')]
            combined.sort_index(inplace=True)
            combined.to_csv(out_bars)
            print(f'Appended {len(ohlcv)} new bars to {out_bars}')
        else:
            ohlcv.to_csv(out_bars, index=False)
            print(f'Wrote {len(ohlcv)} bars to {out_bars}')


if __name__ == '__main__':
    main()
