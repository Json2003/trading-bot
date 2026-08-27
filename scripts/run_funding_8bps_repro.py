#!/usr/bin/env python3
"""Run the unchanged 8-bps hypothesis through a newer completed-data cutoff."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import run_funding_positioning_reversal as impl
except ModuleNotFoundError:
    # GitHub Actions invokes this file as a script, so the repository root is
    # not necessarily importable as the "scripts" package.
    import run_funding_positioning_reversal as impl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--btc-funding-path", type=Path, required=True)
    parser.add_argument("--eth-funding-path", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")).astimezone(timezone.utc)
    if cutoff <= impl.DISCOVERY_END:
        raise ValueError("reproducibility cutoff must be after discovery")
    impl.END = cutoff
    sys.argv = [
        "run_funding_positioning_reversal.py",
        "--btc-path", str(args.btc_path),
        "--eth-path", str(args.eth_path),
        "--btc-funding-path", str(args.btc_funding_path),
        "--eth-funding-path", str(args.eth_funding_path),
        "--output", str(args.output),
    ]
    impl.main()


if __name__ == "__main__":
    main()
