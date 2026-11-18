#!/usr/bin/env python3
"""Convert v3 candidate reports to the common research-result contract."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.result_adapter import canonicalize_v3


def canonicalize_state(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Canonicalize every strategy candidate from every controller report.

    ``last_run.report_paths`` is keyed by order notional (for example,
    ``4000`` and ``6000``), while each referenced report is keyed by strategy
    candidate name.  Keep both dimensions in the governed result key so the
    two independently evaluated notionals are not confused or overwritten.
    """

    report_paths = state.get("last_run", {}).get("report_paths", {})
    if not isinstance(report_paths, dict) or not report_paths:
        raise ValueError("controller state has no report paths")

    output: dict[str, dict[str, Any]] = {}
    for order_notional, raw_path in sorted(
        report_paths.items(), key=lambda item: str(item[0])
    ):
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        candidates = report.get("candidates", {})
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"controller report has no candidates: {path}")
        for candidate_id in sorted(candidates, key=str):
            governed_id = f"{order_notional}:{candidate_id}"
            result = canonicalize_v3(report, str(candidate_id))
            result["candidate_id"] = governed_id
            output[governed_id] = result
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    output = canonicalize_state(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "candidates": output}, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": sorted(output)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
