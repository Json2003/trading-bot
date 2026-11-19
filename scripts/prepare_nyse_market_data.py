#!/usr/bin/env python3
"""Capture completed public daily bars and prepare a frozen NYSE research manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import DecimalException
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from research.nyse_capture import calendar, capture, captured_bars, digest, encode, receipt_index, restore_capture, save_new  # noqa: E402
from research.nyse_preparation import prepare  # noqa: E402
from research.nyse_offline import DataGap, read_json, timestamp  # noqa: E402


DEFAULT_PLAN = ROOT / "research/experiments/nyse_prospective_capture_v1.json"
SPEC_PATH = ROOT / "research/experiments/nyse_momentum_retention_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--data-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture", help="Acquire and timestamp public daily bars.")
    collect.add_argument("--from", dest="start", help="First session. Defaults to the earliest uncovered session.")
    collect.add_argument("--through", help="Last session. Defaults to the latest completed session in the plan.")
    check = commands.add_parser("prepare", help="Validate receipts and build inputs when the full block is ready.")
    check.add_argument("--review", type=Path, required=True)
    recover = commands.add_parser("restore", help="Restore byte-exact raw inputs from a saved receipt index.")
    recover.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw_plan = args.plan.read_bytes()
        plan = read_json(raw_plan)
        plan_hash = digest(raw_plan)
        if plan["schema_version"] != 1:
            raise DataGap("unsupported capture plan version")
        captures = args.data_dir / "captures"
        if args.command == "restore":
            path = restore_capture(read_json(args.index.read_bytes()), plan_hash, captures)
            print(encode({"status": "capture_restored", "file": str(path)}).decode(), end="")
            return 0
        if args.command == "capture":
            now = datetime.now(timezone.utc)
            sessions = [r["session"] for r in calendar(plan)
                        if timestamp(r["close_at"]) <= now and r["session"] <= plan["block"]["end"]]
            if not sessions:
                raise DataGap("the frozen calendar has no completed sessions")
            through = args.through or sessions[-1]
            if through not in sessions:
                raise DataGap("capture end must be a completed session in the frozen window")
            bars, _ = captured_bars(plan, plan_hash, captures)
            start = args.start or next((d for d in sessions if d <= through
                                       and any((s, d) not in bars for s in plan["symbols"])), None)
            if not start:
                print(encode({"status": "already_captured", "through": through}).decode(), end="")
                return 0
            path = capture(plan, plan_hash, start, through, captures)
            bars, _ = captured_bars(plan, plan_hash, captures)
            index = args.data_dir / "indexes" / path.name
            save_new(index, encode(receipt_index(path)))
            missing = [(s, d) for d in sessions if start <= d <= through
                       for s in plan["symbols"] if (s, d) not in bars]
            print(encode({"status": "awaiting_source_publication" if missing else "capture_complete",
                          "file": str(path), "receipt_index": str(index), "from": start, "through": through,
                          "missing_count": len(missing), "first_missing": missing[:12]}).decode(), end="")
            return 2 if missing else 0
        spec = read_json(SPEC_PATH.read_bytes())
        report = prepare(plan, plan_hash, captures, args.review, args.data_dir, spec)
    except (OSError, ValueError, KeyError, TypeError, IndexError, DecimalException) as exc:
        print(encode({"status": "blocked_market_data", "reason": str(exc),
                      "strategy_trials": 0, "expected_profit": None}).decode(), end="")
        return 2
    raw = encode(report)
    record = args.data_dir / "checks" / (digest(raw) + ".json")
    save_new(record, raw)
    print(encode({**report, "check_record": str(record)}).decode(), end="")
    return 0 if report["status"] == "ready_for_offline_replay" else 2


if __name__ == "__main__":
    raise SystemExit(main())
