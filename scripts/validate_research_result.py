#!/usr/bin/env python3
"""Validate the minimum contract for a research result."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

REQUIRED = {"net_return", "drawdown", "sharpe", "profit_factor", "trade_count", "execution_costs"}

def validate(payload: dict) -> None:
    missing = REQUIRED.difference(payload)
    if missing:
        raise ValueError("missing metrics: " + ", ".join(sorted(missing)))
    for key in ("orders_placed", "paper_orders_placed", "leverage_enabled"):
        if payload.get(key, False):
            raise ValueError("research result violates no-execution boundary: " + key)
    if payload.get("promotion_allowed", True):
        raise ValueError("research result must explicitly disable promotion")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("result must be an object")
    validate(payload)
    print("research result contract: valid")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
