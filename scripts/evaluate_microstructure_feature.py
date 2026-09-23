#!/usr/bin/env python3
"""Research-only baseline versus one frozen microstructure feature.

The bar file has decision_ts, next_bar_executable_return, and named numeric
baseline columns. The return must be measured from next-candle entry onward;
this script does not infer fills from close-to-close prices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import hashlib
from datetime import timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.microstructure_feature import align_asof, load_measurements, utc


def read_bars(path: Path, columns: list[str]):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"decision_ts", "next_bar_executable_return", *columns}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"bars missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    times = [utc(row["decision_ts"]) for row in rows]
    if times != sorted(set(times)):
        raise ValueError("decision timestamps must be strictly increasing")
    x = np.asarray([[float(row[key]) for key in columns] for row in rows])
    returns = np.asarray([float(row["next_bar_executable_return"]) for row in rows])
    if not np.isfinite(x).all() or not np.isfinite(returns).all():
        raise ValueError("nonfinite bar values")
    return times, x, returns


def evaluate(bars: Path, measurements: Path, columns: list[str], max_age_hours: float):
    times, baseline, returns = read_bars(bars, columns)
    if len(times) < 100:
        raise ValueError("need at least 100 chronological bars")
    aligned = np.asarray(
        align_asof(times, load_measurements(measurements), max_age=timedelta(hours=max_age_hours))
    )
    available = aligned[:, 1] == 0
    if available.mean() < 0.8:
        raise ValueError(f"feature coverage {available.mean():.1%} is below 80%")
    augmented = np.column_stack((baseline, aligned))
    labels = (returns > 0).astype(int)
    train_end = int(len(times) * 0.6)
    validation_end = int(len(times) * 0.8)
    if len(set(labels[:train_end])) < 2:
        raise ValueError("training labels contain only one class")
    results = {}
    for name, features in (("baseline", baseline), ("candidate", augmented)):
        # Scaling and fitting see only the development slice. The untouched
        # chronological final slice is reported once, never used for selection.
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        model.fit(features[:train_end], labels[:train_end])
        scores = model.predict_proba(features)[:, 1]
        results[name] = {}
        for label, start, end in (("validation", train_end, validation_end), ("holdout", validation_end, len(times))):
            take = scores[start:end] >= 0.55
            net = np.where(take, returns[start:end] - 0.0086, 0.0)
            results[name][label] = {
                "rows": end - start,
                "entries": int(take.sum()),
                "net_return_sum": float(net.sum()),
                "hit_rate": float(np.mean((scores[start:end] >= 0.5) == labels[start:end])),
                "feature_coverage": float(available[start:end].mean()),
                "first_decision_ts": times[start].isoformat(),
                "last_decision_ts": times[end - 1].isoformat(),
            }
    return {
        "status": "research_only_no_promotion",
        "cost_bps_per_entry": 86,
        "entry_probability_threshold": 0.55,
        "train_rows": train_end,
        "baseline_columns": columns,
        "results": results,
        "limitations": [
            "The return column must already reflect next-candle entry and executable fills.",
            "A historical holdout viewed during design is diagnostic, not future independent confirmation.",
            "Rows and entries are not independent walk-forward blocks; no model is eligible for activation here.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--baseline-columns", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    required = {"candidate_id", "source_url", "published_at", "measurement", "formula", "data_source", "symbol", "venue", "max_age_hours"}
    if not isinstance(candidate, dict) or not required.issubset(candidate):
        parser.error(f"candidate JSON requires {sorted(required)}")
    utc(candidate["published_at"])
    max_age = float(candidate["max_age_hours"])
    if not math.isfinite(max_age) or max_age <= 0:
        parser.error("candidate max_age_hours must be positive and finite")
    for key in required - {"max_age_hours"}:
        if not isinstance(candidate[key], str) or not candidate[key].strip():
            parser.error(f"candidate {key} must be a nonempty string")
    result = evaluate(args.bars, args.measurements, args.baseline_columns, max_age)
    result["candidate"] = candidate
    result["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("candidate", args.candidate), ("bars", args.bars), ("measurements", args.measurements))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
