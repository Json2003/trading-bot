#!/usr/bin/env python3
"""Convert v3 candidate reports to the common research-result contract."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from research.result_adapter import canonicalize_v3

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    report_paths = state.get("last_run", {}).get("report_paths", {})
    if not isinstance(report_paths, dict) or not report_paths:
        raise ValueError("controller state has no report paths")
    output = {}
    for candidate_id, raw_path in report_paths.items():
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        output[candidate_id] = canonicalize_v3(report, candidate_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "candidates": output}, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": sorted(output)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
