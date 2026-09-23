#!/usr/bin/env python3
"""Research-only pattern scan with discovery and frozen holdout gates.

This is a cost-referenced gross-movement diagnostic, not an execution
backtest. It uses completed candles and non-overlapping labels. The shared
execution model supplies the stress-cost reference; execution realism is
deferred to the fill-based validation stage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

HORIZONS = (6, 24, 48, 96)
BLOCKS = 6
MIN_PER_BLOCK = 20
HOLDOUT_MIN_HOURS = BLOCKS * MIN_PER_BLOCK * max(HORIZONS)
STRESS_COST = STRESS_EXECUTION.round_trip_bps / 10_000.0
PATTERNS = (
    "trend_structure", "breakout", "vol_expansion",
    "vol_contraction_break", "relative_strength", "momentum_decay",
    "mean_reversion", "exhaustion",
)
FAMILY_SIZE = len(PATTERNS) * len(HORIZONS)


def mean(values):
    return sum(values) / len(values) if values else math.nan


def feat(bars):
    close = [float(x.close) for x in bars]
    high = [float(x.high) for x in bars]
    low = [float(x.low) for x in bars]
    candle_range = [
        (high[i] - low[i]) / close[i] if close[i] else math.nan
        for i in range(len(close))
    ]
    out = {"c": close, "h": high, "l": low, "r": candle_range}
    for n in (6, 24, 48, 96, 200):
        out[f"m{n}"] = [
            math.nan if i < n else close[i] / close[i - n] - 1
            for i in range(len(close))
        ]
        out[f"avg_r{n}"] = [
            mean(candle_range[i - n + 1:i + 1])
            if i >= n - 1 else math.nan
            for i in range(len(close))
        ]
        out[f"avg_c{n}"] = [
            mean(close[i - n + 1:i + 1])
            if i >= n - 1 else math.nan
            for i in range(len(close))
        ]
    out["high48"] = [
        max(high[i - 48:i]) if i >= 49 else math.nan
        for i in range(len(close))
    ]
    out["low48"] = [
        min(low[i - 48:i]) if i >= 49 else math.nan
        for i in range(len(close))
    ]
    return out


def ok(*values):
    return all(math.isfinite(float(value)) for value in values)


def choose(name, i, b, e):
    assets = (("BTC", b), ("ETH", e))
    if i < 220:
        return None, 0
    if name == "relative_strength":
        if not ok(b["m24"][i], e["m24"][i]):
            return None, 0
        if b["m24"][i] > e["m24"][i] + 0.01:
            return "BTC", 1
        if e["m24"][i] > b["m24"][i] + 0.01:
            return "ETH", 1
        return None, 0

    eligible = []
    for symbol, data in assets:
        if not ok(data["c"][i], data["m24"][i], data["m48"][i],
                  data["m96"][i], data["avg_r24"][i]):
            continue
        trend = data["c"][i] > data["avg_c200"][i] and data["m24"][i] > 0
        breakout = (
            ok(data["high48"][i], data["low48"][i])
            and (data["c"][i] > data["high48"][i]
                 or data["c"][i] < data["low48"][i])
        )
        expansion = data["r"][i] > 1.25 * data["avg_r24"][i]
        contraction = data["r"][i] < 0.75 * data["avg_r24"][i]

        if name == "trend_structure" and trend and data["m48"][i] > data["m24"][i]:
            eligible.append((data["m24"][i], symbol, 1))
        elif name == "breakout" and breakout and trend:
            direction = 1 if data["m24"][i] > 0 else -1
            eligible.append((abs(data["m24"][i]), symbol, direction))
        elif name == "vol_expansion" and expansion and trend:
            eligible.append((data["m24"][i], symbol, 1))
        elif name == "vol_contraction_break" and contraction and breakout:
            direction = 1 if data["m24"][i] > 0 else -1
            eligible.append((abs(data["m24"][i]), symbol, direction))
        elif name == "momentum_decay" and trend and data["m48"][i] > 0 and data["m24"][i] < data["m48"][i] * 0.5:
            eligible.append((data["m24"][i], symbol, 1))
        elif name == "mean_reversion" and data["m24"][i] < -0.03:
            eligible.append((abs(data["m24"][i]), symbol, 1))
        elif name == "exhaustion" and data["m48"][i] > 0.15 and data["m24"][i] < 0:
            eligible.append((abs(data["m24"][i]), symbol, -1))

    if not eligible:
        return None, 0
    _, symbol, direction = max(eligible)
    return symbol, direction


def summarize(rows, segment_start, segment_end):
    groups = [[] for _ in range(BLOCKS)]
    block_size = (segment_end - segment_start) / BLOCKS
    for index, value in rows:
        block = min(
            BLOCKS - 1,
            int((index - segment_start) / block_size),
        )
        groups[block].append(value)

    counts = [len(group) for group in groups]
    means = [mean(group) if group else None for group in groups]
    valid = [value for value in means if value is not None]
    med = median(valid) if valid else math.nan
    stress_multiple = med / STRESS_COST if math.isfinite(med) else math.nan
    gate = (
        len(rows) >= BLOCKS * MIN_PER_BLOCK
        and all(count >= MIN_PER_BLOCK for count in counts)
        and len(valid) == BLOCKS
        and all(value > 0 for value in valid)
        and stress_multiple >= 4.0
    )
    return {
        "observations": len(rows),
        "block_observations": counts,
        "non_overlapping": True,
        "block_mean_gross_returns": means,
        "median_block_gross_return": med if math.isfinite(med) else None,
        "stress_cost_multiple": (
            stress_multiple if math.isfinite(stress_multiple) else None
        ),
        "passes_4x_stress_gate": gate,
    }


def scan(btc_path: Path, eth_path: Path):
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    required = 3 * 365 * 24
    if len(pair) < required:
        raise ValueError("three years of aligned hourly candles required")
    pair = pair[-required:]
    holdout_start = len(pair) - HOLDOUT_MIN_HOURS
    if holdout_start <= 220:
        raise ValueError("three-year window is too short for the frozen holdout")

    bars = {"BTC": [x.btc for x in pair], "ETH": [x.eth for x in pair]}
    f = {symbol: feat(values) for symbol, values in bars.items()}
    results = {}

    for name in PATTERNS:
        results[name] = {}
        for horizon in HORIZONS:
            discovery = []
            for i in range(220, holdout_start - horizon, horizon):
                symbol, direction = choose(name, i, f["BTC"], f["ETH"])
                if symbol:
                    x = f[symbol]
                    discovery.append(
                        (i, direction * (x["c"][i + horizon] / x["c"][i] - 1))
                    )
            holdout = []
            for i in range(holdout_start, len(pair) - horizon, horizon):
                symbol, direction = choose(name, i, f["BTC"], f["ETH"])
                if symbol:
                    x = f[symbol]
                    holdout.append(
                        (i, direction * (x["c"][i + horizon] / x["c"][i] - 1))
                    )

            discovery_result = summarize(discovery, 0, holdout_start)
            holdout_result = summarize(
                holdout, holdout_start, len(pair)
            )
            results[name][str(horizon)] = {
                "candidate_id": f"{name}:{horizon}",
                "discovery": discovery_result,
                "holdout": holdout_result,
                "selected_for_holdout": discovery_result["passes_4x_stress_gate"],
                "passes_discovery_and_holdout": (
                    discovery_result["passes_4x_stress_gate"]
                    and holdout_result["passes_4x_stress_gate"]
                ),
            }

    passing = [
        f"{name}:{horizon}"
        for name, horizons in results.items()
        for horizon, result in horizons.items()
        if result["passes_discovery_and_holdout"]
    ]
    counts = {
        "robust_candidate": len(passing),
        "discovery_only": sum(
            1 for horizons in results.values()
            for result in horizons.values()
            if result["discovery"]["passes_4x_stress_gate"]
            and not result["holdout"]["passes_4x_stress_gate"]
        ),
        "no_discovery_pass": sum(
            1 for horizons in results.values()
            for result in horizons.values()
            if not result["discovery"]["passes_4x_stress_gate"]
        ),
    }
    return {
        "schema_version": 2,
        "research_only": True,
        "orders_placed": False,
        "leverage_enabled": False,
        "execution_realism": "gross_movement_diagnostic",
        "execution_model_reference": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": pair[0].timestamp.isoformat(),
            "end": pair[-1].timestamp.isoformat(),
            "bars": len(pair),
            "blocks_per_segment": BLOCKS,
            "discovery_end_index": holdout_start - 1,
            "holdout_start_index": holdout_start,
            "holdout_hours": HOLDOUT_MIN_HOURS,
        },
        "costs": {
            "stress_round_trip": STRESS_COST,
            "stress_round_trip_bps": STRESS_EXECUTION.round_trip_bps,
        },
        "method": {
            "horizons_hours": HORIZONS,
            "sample_every_horizon": True,
            "minimum_observations_per_block": MIN_PER_BLOCK,
            "promotion_requires_discovery_and_holdout": True,
            "holdout_is_not_used_for_selection": True,
            "multiple_testing": {
                "family_size": FAMILY_SIZE,
                "correction": "chronological_holdout_confirmation",
                "every_pattern_horizon_requires_confirmation": True,
            },
        },
        "patterns": results,
        "passing_patterns": passing,
        "classification_counts": counts,
        "status": "candidate_found" if passing else "no_candidate_passed",
        "status_note": (
            "A passing gross diagnostic is not profitability evidence; "
            "fill-based net P&L is still required."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = scan(args.btc_path, args.eth_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "passing_patterns": report["passing_patterns"],
    }, indent=2))


if __name__ == "__main__":
    main()
