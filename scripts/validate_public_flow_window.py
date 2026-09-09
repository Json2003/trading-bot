#!/usr/bin/env python3
"""Validate a completed, timestamped BTC/ETH public-flow research window."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def inspect(path: Path, symbol: str, frequency: str = "1min") -> dict:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"timestamp", "completed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{path} timestamps are duplicated or unordered")
    completed = frame["completed"].astype(bool)
    if not completed.all():
        raise ValueError(f"{path} contains incomplete observations")
    expected = pd.date_range(frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1], freq=frequency, tz="UTC")
    if len(expected) != len(frame) or not frame["timestamp"].equals(pd.Series(expected)):
        raise ValueError(f"{path} has gaps in completed flow observations")
    return {"symbol": symbol, "rows": int(len(frame)), "from": frame["timestamp"].iloc[0].isoformat(), "through": frame["timestamp"].iloc[-1].isoformat(), "path": str(path)}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc", type=Path, required=True)
    parser.add_argument("--eth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    btc, eth = inspect(args.btc, "BTCUSDT"), inspect(args.eth, "ETHUSDT")
    if btc["from"] != eth["from"] or btc["through"] != eth["through"] or btc["rows"] != eth["rows"]:
        raise ValueError("BTC and ETH public-flow windows are not aligned")
    report = {"schema_version": 1, "status": "valid", "frequency": "1min", "symbols": [btc, eth], "completed_only": True, "continuous": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
