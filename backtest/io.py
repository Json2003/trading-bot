"""Data IO utilities for backtesting.

Implements:
- load_csv(path): read OHLCV CSV and parse timestamp
- fetch_ccxt(exchange, symbol, timeframe, since, until): fetch OHLCV via CCXT
  with built-in pagination to cover multi-year ranges.

Notes:
- Uses safe third-party imports to avoid local pandas stub shadowing.
"""
from __future__ import annotations

import os, sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _tp(mod_name: str):
    """Import a third-party module robustly.

    Try normal import first. If it resolves to a repo-local shadow or fails,
    retry with site-packages prioritized.
    """
    import importlib, sys as _sys
    # First attempt: normal import
    try:
        mod = importlib.import_module(mod_name)
        src = getattr(mod, "__file__", "") or ""
        if src and REPO_ROOT in os.path.abspath(src):
            raise ImportError("shadowed by repo-local file")
        return mod
    except Exception:
        pass
    # Fallback: prioritize site/dist-packages
    original = _sys.path.copy()
    try:
        site = [p for p in original if ("site-packages" in (p or "")) or ("dist-packages" in (p or ""))]
        rest = [p for p in original if p not in site]
        _sys.path[:] = site + rest
        if mod_name in _sys.modules:
            del _sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        return mod
    finally:
        _sys.path[:] = original


def load_csv(path: str):
    pd = _tp('pandas')
    df = pd.read_csv(path)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    except Exception:
        df["timestamp"] = pd.to_datetime(df["timestamp"]) 
    return df[["timestamp","open","high","low","close","volume"]]


def fetch_ccxt(exchange: str, symbol: str, timeframe: str, since: str, until: str):
    # Use guarded imports that prioritize site-packages
    pd = _tp('pandas')
    dparser = _tp('dateutil.parser')
    _ = _tp('requests')  # ensure real requests is loaded
    ccxt = _tp('ccxt')

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(dparser.parse(since)).timestamp() * 1000)
    until_ms = int(pd.Timestamp(dparser.parse(until)).timestamp() * 1000)

    limit = 1000
    cursor = since_ms
    rows = []
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

    if not rows:
        raise RuntimeError("No OHLCV fetched from exchange")

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if until_ms:
        df = df[df["timestamp"] <= pd.to_datetime(until_ms, unit='ms')]
    return df.reset_index(drop=True)
