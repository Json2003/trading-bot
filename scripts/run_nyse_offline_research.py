#!/usr/bin/env python3
"""Run the frozen NYSE comparison from local hashed files.

Example:
    python scripts/run_nyse_offline_research.py --manifest data/nyse/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from decimal import DecimalException

ROOT = Path(__file__).resolve().parents[1]
# Append the project path so unrelated imports retain their existing precedence.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from research.engine import research_safety_flags  # noqa: E402
from research.nyse_offline import evaluate, load_dataset, read_json, validate_spec  # noqa: E402


def encoded(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def owner_summary(report: dict) -> str:
    lines = [f"**NYSE research run — {report['status']}**", "",
             "Research cash is $5,000, with whole shares, a $1,000 entry cap and 43 bps per traded side.", ""]
    if report.get("data_kind") == "synthetic":
        lines.extend(["This run uses synthetic inputs to verify software behavior.", ""])
    if not report["results"]:
        lines.extend(["Market evaluation is blocked. Performance metrics are unmeasured.", ""])
    else:
        lines.extend(["| Block and arm | Net return | Drawdown | Stress costs | Independent observations |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for row in report["results"]:
            lines.append(f"| {row['block']} / {row['arm']} | {row['net_return']:.2%} | {row['drawdown']:.2%} | "
                         f"${row['execution_costs']:.2f} | {row['independent_observations']} / 20 |")
        lines.append("")
        for row in report["results"]:
            state = row["equity_curve"][-1]
            holdings = ", ".join(f"{sid} × {pos['quantity']}" for sid, pos in state["positions"].items()) or "cash and claims"
            lines.append(f"{row['block']} / {row['arm']} ends with {holdings}. Settled cash is "
                         f"${float(state['settled_cash']):.2f}, receivables ${float(state['receivable_value']):.2f} "
                         f"and fractional interests ${float(state['fraction_value']):.2f}.")
            lines.append("")
            if row["failure_reasons"]:
                lines.append("Review findings are " + ", ".join(row["failure_reasons"]) + ".")
                lines.append("")
    lines.extend(["Evidence requiring attention", ""])
    lines.extend(f"- {reason}" for reason in report.get("failure_reasons", []))
    lines.extend(["", "The complete ledger, rankings, intended orders and metric definitions are in [report.json](report.json).",
                  "Input identity and revisions are retained in [manifest.json](manifest.json) and [spec.json](spec.json).",
                  "", f"Run identity is {report['run_id']}.", ""])
    return "\n".join(lines)


def run(manifest_path: Path, spec_path: Path, output_dir: Path) -> tuple[Path, dict]:
    input_bytes = {}
    report = {"schema_version": 1, **research_safety_flags(), "status": "blocked_market_data",
              "investment_trials": 0, "model_training_runs": 0, "checkpoint_advanced": False,
              "forward_confirmation": False, "eligible_for_confirmation": False, "results": [],
              "metrics": {k: None for k in ("net_return", "drawdown", "sharpe", "profit_factor", "trade_count", "execution_costs")}}
    input_hashes = {}
    try:
        for label, path in (("manifest.json", manifest_path), ("spec.json", spec_path)):
            input_bytes[label] = path.read_bytes()
            input_hashes[label] = hashlib.sha256(input_bytes[label]).hexdigest()
        spec = read_json(input_bytes["spec.json"])
        validate_spec(spec)
        data = load_dataset(manifest_path, spec, manifest_raw=input_bytes["manifest.json"])
        input_hashes.update(data.fingerprints)
        evaluation = evaluate(data, spec)
        input_paths = {name: str((manifest_path.parent / data.manifest["files"][name]["path"]).resolve())
                       for name in data.fingerprints}
        report.update(evaluation)
        report["input_manifest_path"] = str(manifest_path.resolve())
        report["input_file_paths"] = input_paths
        report.pop("metrics")
    except (OSError, ValueError, KeyError, TypeError, IndexError, DecimalException, OverflowError) as exc:
        report["failure_reasons"] = [f"{type(exc).__name__}: {exc}"]
    sources = [Path(__file__), ROOT / "research/nyse_accounting.py", ROOT / "research/nyse_offline.py",
               ROOT / "research/engine.py"]
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    identity = {"inputs": input_hashes, "code": code_hashes}
    if len(input_bytes) != 2:
        identity["missing_input_paths"] = [str(manifest_path.resolve()), str(spec_path.resolve())]
    run_id = hashlib.sha256(encoded(identity)).hexdigest()
    destination = output_dir / run_id
    if destination.exists():
        previous = read_json((destination / "report.json").read_bytes())
        if previous["run_id"] != run_id or previous["identity"] != identity:
            raise ValueError("existing research artifact identity differs")
        return destination, previous
    report.update({"run_id": run_id, "identity": identity,
                   "generated_at": datetime.now(timezone.utc).isoformat(),
                   "source_commit": os.environ.get("RESEARCH_SOURCE_COMMIT", "local-source-hashes")})
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nyse-", dir=output_dir))
    try:
        for name, raw in input_bytes.items():
            (staging / name).write_bytes(raw)
        (staging / "report.json").write_bytes(encoded(report))
        summary = {k: v for k, v in report.items() if k != "results"}
        summary["results"] = [{k: v for k, v in row.items() if k not in {"decisions", "intentions", "ledger", "realizations", "equity_curve"}}
                              for row in report["results"]]
        (staging / "summary.json").write_bytes(encoded(summary))
        (staging / "summary.md").write_text(owner_summary(report), encoding="utf-8")
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            previous = read_json((destination / "report.json").read_bytes())
            if previous["identity"] != identity:
                raise ValueError("concurrent research artifact identity differs")
            report = previous
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=ROOT / "research/experiments/nyse_momentum_retention_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/nyse-offline")
    args = parser.parse_args()
    directory, report = run(args.manifest, args.spec, args.output_dir)
    print(json.dumps({"status": report["status"], "run_id": report["run_id"],
                      "report": str(directory / "report.json"), "summary": str(directory / "summary.json"),
                      "review": str(directory / "summary.md"),
                      "failure_reasons": report.get("failure_reasons", [])}, indent=2))
    return 2 if report["status"] == "blocked_market_data" else 0


if __name__ == "__main__":
    raise SystemExit(main())
