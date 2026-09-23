#!/usr/bin/env python3
"""Fail-closed validation of the frozen high_quality_entry candidate.

The candidate and parameters are read from the pre-registered definitions.
Windows never overlap, the final holdout is evaluated last, and missing
trade-level or exact-fill metrics block advancement.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from typing import Mapping

try:
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair, build_pair_features, run_pair
    from scripts.run_v3_exploration import candidate_definitions
except ModuleNotFoundError:
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair, build_pair_features, run_pair
    from run_v3_exploration import candidate_definitions

def safe(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    return v

def result(pair, features, config, start, end, fees, slip):
    r = run_pair(pair, initial_balance=75000.0, order_notional=6000.0,
                 fees_bps=fees, slippage_bps=slip, config=config,
                 start_index=start, end_index=end, feature_map=features)
    # The current v3 engine does not expose a fill ledger/equity series.
    r.update({
        "sharpe_annualized": None,
        "profit_factor": None,
        "exact_fill_ledger": False,
        "trade_level_metrics_complete": False,
    })
    return r

def run_validation(btc_path: Path, eth_path: Path, output: Path):
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    if len(pair) < 3 * 365 * 24:
        raise ValueError("need at least three years of aligned hourly candles")
    pair = pair[-3 * 365 * 24:]
    n = len(pair)
    blocks = {
        "development": (0, n // 2),
        "walk_forward_1": (n // 2, n * 5 // 8),
        "walk_forward_2": (n * 5 // 8, n * 3 // 4),
        "walk_forward_3": (n * 3 // 4, n * 7 // 8),
        "final_unseen_holdout": (n * 7 // 8, n),
    }
    config = candidate_definitions()["high_quality_entry"]
    features = build_pair_features(pair, config)
    reports = {}
    for name, (start, end) in blocks.items():
        reports[name] = {
            "indices": {"start": start, "end": end, "bars": end - start},
            "base": result(pair, features, config, start, end, 10.0, 9.0),
            "stress": result(pair, features, config, start, end, 20.0, 23.0),
        }
    test_blocks = [reports[f"walk_forward_{i}"] for i in range(1, 4)]
    base_returns = [float(x["base"]["return_pct"]) for x in test_blocks]
    stress_returns = [float(x["stress"]["return_pct"]) for x in test_blocks]
    reasons = []
    if not all(x > 0 for x in base_returns):
        reasons.append("not all base walk-forward blocks are positive")
    if not all(x > 0 for x in stress_returns):
        reasons.append("not all stress walk-forward blocks are positive")
    for name, report in reports.items():
        for scenario in ("base", "stress"):
            r = report[scenario]
            if r["entries"] < 5:
                reasons.append(f"{name} {scenario} has fewer than five entries")
            if r["sharpe_annualized"] is None:
                reasons.append(f"{name} {scenario} Sharpe unavailable")
            if r["profit_factor"] is None:
                reasons.append(f"{name} {scenario} profit factor unavailable")
            if not r["exact_fill_ledger"]:
                reasons.append(f"{name} {scenario} exact fill ledger unavailable")
    report = {
        "schema_version": 1,
        "research_only": True,
        "candidate": "high_quality_entry",
        "parameters": config.as_dict(),
        "window": {"start": pair[0].timestamp.isoformat(), "end": pair[-1].timestamp.isoformat(), "bars": n, "completed_candles_only": True},
        "blocks": reports,
        "non_overlapping": True,
        "forward_test_reused_for_training": False,
        "median_walk_forward_return_pct": {"base": statistics.median(base_returns), "stress": statistics.median(stress_returns)},
        "eligible_for_confirmation": not reasons,
        "failure_reasons": reasons,
        "promotion_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe(report), indent=2, allow_nan=False), encoding="utf-8")
    return report

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--btc-path", type=Path, required=True)
    p.add_argument("--eth-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run_validation(args.btc_path, args.eth_path, args.output)
    print(json.dumps({"eligible_for_confirmation": report["eligible_for_confirmation"], "failure_reasons": report["failure_reasons"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
