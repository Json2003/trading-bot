#!/usr/bin/env python3
"""Apply the research result contract and advance a data checkpoint.

The controller emits a nested, multi-notional report. This wrapper accepts
either that canonical bundle or one legacy flat result, validates every
candidate before advancing the checkpoint, and writes a single governed
report.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.checkpoint import (
    CheckpointError,
    advance_checkpoint,
    atomic_write_json,
    load_checkpoint,
    sha256_file,
)

REQUIRED = {
    "net_return",
    "drawdown",
    "sharpe",
    "profit_factor",
    "trade_count",
    "execution_costs",
}
SAFETY_FALSE = (
    "orders_placed",
    "paper_orders_placed",
    "leverage_enabled",
    "risk_limits_changed",
)


def validate_result(payload: Mapping[str, Any], label: str = "result") -> None:
    missing = REQUIRED.difference(payload)
    if missing:
        raise CheckpointError(
            f"{label} missing metrics: " + ", ".join(sorted(missing))
        )
    for key in SAFETY_FALSE:
        if payload.get(key, False):
            raise CheckpointError(f"{label} violates research boundary: {key}")
    if payload.get("promotion_allowed", True):
        raise CheckpointError(f"{label} must explicitly disable promotion")


def iter_results(payload: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    candidates = payload.get("candidates")
    if candidates is None:
        yield "result", payload
        return
    if not isinstance(candidates, Mapping) or not candidates:
        raise CheckpointError("canonical result bundle must contain candidates")
    for candidate_id, candidate in sorted(candidates.items(), key=lambda item: str(item[0])):
        if not isinstance(candidate, Mapping):
            raise CheckpointError(f"candidate {candidate_id} must be an object")
        yield str(candidate_id), candidate


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
        report = {
            "schema_version": 1,
            "status": "skip",
            "reason": "no_new_completed_data_after_checkpoint",
            "previous_checkpoint": prior,
            "completed_through": current,
            "candidate_results": [],
            "checkpoint_advanced": False,
            "research_only": True,
            "orders_placed": False,
            "paper_orders_placed": False,
            "leverage_enabled": False,
            "risk_limits_changed": False,
            "promotion_allowed": False,
        }
        atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    for path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise CheckpointError(f"{path} must contain a JSON object")
        for candidate_id, candidate in iter_results(payload):
            validate_result(candidate, f"{path}:{candidate_id}")
            results.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "candidate_id": candidate_id,
                    "result": dict(candidate),
                }
            )
    if not results:
        raise CheckpointError("no candidate results were supplied")

    next_checkpoint = advance_checkpoint(
        checkpoint,
        manifest,
        evaluated_through=current,
    )
    atomic_write_json(args.checkpoint, next_checkpoint)
    report = {
        "schema_version": 1,
        "status": "evaluated",
        "dataset_id": manifest["dataset_id"],
        "completed_through": current,
        "previous_checkpoint": prior,
        "candidate_results": results,
        "checkpoint_advanced": True,
        "research_only": True,
        "orders_placed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
        "risk_limits_changed": False,
        "promotion_allowed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_count": len(results),
                "checkpoint_advanced": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
