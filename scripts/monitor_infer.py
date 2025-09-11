#!/usr/bin/env python3
from __future__ import annotations

"""Simple monitoring loop for the inference API.

Probes /health and /infer periodically and appends JSONL to artifacts/ml_monitor.log.

Usage examples:
  python scripts/monitor_infer.py --interval 30 --threshold 0.55
  python scripts/monitor_infer.py --symbol BTC/USDT --timeframe 4h --loops 5

Defaults:
  - URL: http://127.0.0.1:8000
  - If --symbol/--timeframe not provided, reads active tag from models/active_tag.txt
    and infers (symbol,timeframe) from models/registry.json.
  - Uses local feature store data (CSV/parquet). Set FEATURE_CACHE=1 to speed load.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure repo root on sys.path and avoid local shadows
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if p not in ('', REPO_ROOT)] + [REPO_ROOT]

# Avoid local requests.py by using urllib
import urllib.request
import ssl


def load_active_symbol_tf() -> Tuple[str, str]:
    tag_path = Path("models/active_tag.txt")
    reg_path = Path("models/registry.json")
    tag = tag_path.read_text().strip()
    with reg_path.open() as f:
        reg = json.load(f)
    meta = reg.get(tag) or {}
    sym = meta.get("symbol") or "BTC/USDT"
    tf = meta.get("timeframe") or "4h"
    return sym, tf


def fetch_features(symbol: str, timeframe: str, lookback: int = 64):
    from data.feature_store import get_supervised_dataset
    # Use any wide range; take last 64 rows for the request
    X, _ = get_supervised_dataset(symbol, timeframe, "2021-01-01", "2024-01-01", lookback, 1, cache=True, source="local")
    seq = X.tail(lookback).values.astype(float).tolist()
    return seq


def post_json(url: str, payload: dict, timeout: float = 3.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_json(url: str, timeout: float = 3.0) -> dict:
    req = urllib.request.Request(url)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--loops", type=int, default=-1, help="-1 = infinite")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    # Resolve symbol/timeframe if not given
    symbol = args.symbol
    timeframe = args.timeframe
    if not (symbol and timeframe):
        s, t = load_active_symbol_tf()
        symbol = symbol or s
        timeframe = timeframe or t

    log_dir = Path("artifacts"); log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ml_monitor.log"

    n = 0
    while True:
        n += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # /health
        try:
            health = get_json(args.url.rstrip("/") + "/health")
        except Exception as e:
            health = {"error": str(e)}

        # /infer
        try:
            seq = fetch_features(symbol, timeframe, 64)
            infer = post_json(args.url.rstrip("/") + "/infer", {"features": seq, "threshold": args.threshold})
        except Exception as e:
            infer = {"error": str(e)}

        # Append logs
        rec = {"ts": ts, "health": health, "infer": infer, "symbol": symbol, "timeframe": timeframe}
        try:
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

        # Print one line summary
        print(json.dumps(rec))

        if args.loops > 0 and n >= args.loops:
            break
        time.sleep(max(0.1, float(args.interval)))


if __name__ == "__main__":
    main()
