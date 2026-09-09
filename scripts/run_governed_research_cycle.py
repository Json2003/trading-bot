#!/usr/bin/env python3
"""Apply the research result contract and advance a data checkpoint."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from research.checkpoint import CheckpointError, advance_checkpoint, atomic_write_json, load_checkpoint, sha256_file

REQUIRED = {"net_return", "drawdown", "sharpe", "profit_factor", "trade_count", "execution_costs"}

def validate_result(payload: dict) -> None:
    missing = REQUIRED.difference(payload)
    if missing:
        raise CheckpointError("result missing metrics: " + ", ".join(sorted(missing)))
    for key in ("orders_placed", "paper_orders_placed", "leverage_enabled"):
        if payload.get(key, False):
            raise CheckpointError("research boundary violated: " + key)
    if payload.get("promotion_allowed", True):
        raise CheckpointError("promotion must be disabled")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(args.checkpoint)
    prior = checkpoint.get("last_completed_candle")
    current = str(manifest["completed_through"])
    if prior is not None and current <= str(prior):
        report = {"schema_version": 1, "status": "skip", "reason": "no_new_completed_data_after_checkpoint", "previous_checkpoint": prior, "completed_through": current, "candidate_results": [], "research_only": True, "orders_placed": False, "leverage_enabled": False, "promotion_allowed": False}
        atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0
    results = []
    for path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_result(payload)
        results.append({"path": str(path), "sha256": sha256_file(path), "result": payload})
    next_checkpoint = advance_checkpoint(checkpoint, manifest, evaluated_through=current)
    atomic_write_json(args.checkpoint, next_checkpoint)
    report = {"schema_version": 1, "status": "evaluated", "dataset_id": manifest["dataset_id"], "completed_through": current, "previous_checkpoint": prior, "candidate_results": results, "checkpoint_advanced": True, "research_only": True, "orders_placed": False, "leverage_enabled": False, "promotion_allowed": False, "generated_at": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(args.output, report)
    print(json.dumps({"status": report["status"], "checkpoint_advanced": True}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
