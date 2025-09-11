#!/usr/bin/env python3
from __future__ import annotations

"""Fetch paginated OHLCV from KuCoin (via CCXT) and save to CSV/Parquet.

Examples:
  python scripts/fetch_kucoin_ohlcv.py --symbol BTC/USDT --timeframe 4h \
    --since 2021-01-01 --until 2024-01-01 --out data/BTC_USDT_4h.csv

Notes:
  - Timestamps are saved as milliseconds since epoch (like sample_ohlcv.csv).
  - Also writes a Parquet alongside if pyarrow/fastparquet is installed.
  - Index/timezone handling: downstream loaders convert to UTC DatetimeIndex.
"""

import os
import sys
import argparse


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if p not in ('', REPO_ROOT)] + [REPO_ROOT]


def _import_site(mod_name: str):
    import importlib
    mod = sys.modules.get(mod_name)
    if mod is not None:
        src = getattr(mod, "__file__", "") or ""
        try:
            if REPO_ROOT in os.path.abspath(src):
                del sys.modules[mod_name]
        except Exception:
            pass
    original = sys.path.copy()
    try:
        repo_paths = {p for p in original if REPO_ROOT in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        return importlib.import_module(mod_name)
    finally:
        sys.path = original


def fetch_ohlcv(exchange: str, symbol: str, timeframe: str, since: str, until: str):
    pd = _import_site('pandas')
    dparser = _import_site('dateutil.parser')
    _import_site('requests')
    ccxt = _import_site('ccxt')

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(dparser.parse(since)).timestamp() * 1000) if since else None
    until_ms = int(pd.Timestamp(dparser.parse(until)).timestamp() * 1000) if until else None
    limit = 1000
    rows = []
    cursor = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
        if until_ms and cursor >= until_ms:
            break
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]) if rows else pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    if not df.empty:
        # keep ms epoch int for CSV compatibility; parquet will store datetime if converted by downstream
        pass
    return df


def main():
    ap = argparse.ArgumentParser(description="Fetch KuCoin OHLCV and save to CSV/Parquet")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--since", required=True, help="e.g., 2021-01-01")
    ap.add_argument("--until", required=True, help="e.g., 2024-01-01")
    ap.add_argument("--out", default=None, help="Output CSV path (default data/<SYMBOL>_<TF>.csv)")
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        sym = args.symbol.replace('/', '_')
        out_path = os.path.join(REPO_ROOT, "data", f"{sym}_{args.timeframe}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = fetch_ohlcv("kucoin", args.symbol, args.timeframe, args.since, args.until)
    if df.empty:
        print("No data fetched.")
        return
    df.to_csv(out_path, index=False)
    # Try to write parquet alongside
    try:
        pq_path = os.path.splitext(out_path)[0] + ".parquet"
        pd = _import_site('pandas')
        dft = df.copy()
        dft['timestamp'] = pd.to_datetime(dft['timestamp'], unit='ms', utc=True)
        dft = dft.set_index('timestamp')
        dft.to_parquet(pq_path)
        print(f"Saved CSV: {out_path} and Parquet: {pq_path}")
    except Exception:
        print(f"Saved CSV: {out_path} (parquet skipped)")


if __name__ == "__main__":
    main()

