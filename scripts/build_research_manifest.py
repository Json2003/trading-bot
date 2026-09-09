#!/usr/bin/env python3
"""Build a checksummed manifest for completed normalized hourly data."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from research.checkpoint import atomic_write_json, sha256_file

def inspect_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["timestamp"])
    if frame.empty:
        raise ValueError(f"{path} is empty")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path} timestamps are not ordered")
    expected = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq="1h", tz="UTC")
    if len(expected) != len(timestamps) or not timestamps.equals(pd.Series(expected)):
        raise ValueError(f"{path} has gaps in completed hourly candles")
    return {"path": str(path), "sha256": sha256_file(path), "rows": int(len(frame)), "from": timestamps.iloc[0].isoformat(), "through": timestamps.iloc[-1].isoformat()}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, action="append", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [inspect_csv(path) for path in args.file]
    manifest = {"schema_version": 1, "dataset_id": args.dataset_id, "generated_at": datetime.now(timezone.utc).isoformat(), "completed_through": min(item["through"] for item in files), "frequency": "1h", "completed_candles_only": True, "files": files}
    atomic_write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
