#!/usr/bin/env python3
"""Pre-registered, cost-referenced gross-movement diagnostic.

This screen is intentionally not an execution backtest. It uses completed
candles and non-overlapping labels to discover directional movement, then
requires the same frozen candidate/horizon to pass a chronological holdout.
Execution costs are read from the shared execution model for the reference
gate; latency, fills, funding, and rejected orders are not simulated here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

HORIZONS = (6, 12, 24, 48, 96)
BLOCKS = 6
MIN_OBSERVATIONS_PER_BLOCK = 20
HOLDOUT_MIN_HOURS = BLOCKS * MIN_OBSERVATIONS_PER_BLOCK * max(HORIZONS)
STRESS_COST = STRESS_EXECUTION.round_trip_bps / 10_000.0
FAMILY_SIZE = 3 * len(HORIZONS)


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def rolling_mean(values: list[float], window: int, i: int) -> float:
    if i < window - 1:
        return math.nan
    return sum(values[i - window + 1:i + 1]) / window


def features(bars: list[Any]) -> dict[str, list[float]]:
    close = [float(bar.close) for bar in bars]
    high = [float(bar.high) for bar in bars]
    low = [float(bar.low) for bar in bars]
    candle_range = [(h - l) / c if c else math.nan
                    for h, l, c in zip(high, low, close)]
    mom24 = [math.nan if i < 24 else close[i] / close[i - 24] - 1
             for i in range(len(close))]
    sma200 = [rolling_mean(close, 200, i) for i in range(len(close))]
    range24 = [rolling_mean(candle_range, 24, i)
               for i in range(len(close))]
    prior_high48 = [max(high[i - 48:i]) if i >= 49 else math.nan
                    for i in range(len(close))]
    return {
        "close": close, "mom24": mom24, "sma200": sma200,
        "range": candle_range, "range24": range24,
        "prior_high48": prior_high48,
    }


def signal_names() -> tuple[str, ...]:
    return ("trend_volatility", "breakout_volatility",
            "relative_strength_volatility")


def choose_signal(
    name: str,
    i: int,
    btc: dict[str, list[float]],
    eth: dict[str, list[float]],
) -> tuple[str | None, int]:
    if i < 220:
        return None, 0
    candidates = (("BTC", btc), ("ETH", eth))
    if name == "relative_strength_volatility":
        if not finite(btc["mom24"][i], eth["mom24"][i]):
            return None, 0
        leader = "BTC" if btc["mom24"][i] >= eth["mom24"][i] else "ETH"
        lead, lag = (btc, eth) if leader == "BTC" else (eth, btc)
        if not finite(lead["range"][i], lead["range24"][i],
                      lead["sma200"][i], lag["mom24"][i]):
            return None, 0
        if (lead["mom24"][i] - lag["mom24"][i] >= 0.01
                and lead["close"][i] > lead["sma200"][i]
                and lead["range"][i] >= 1.25 * lead["range24"][i]):
            return leader, 1
        return None, 0

    eligible = []
    for symbol, data in candidates:
        if not finite(data["mom24"][i], data["sma200"][i],
                      data["range"][i], data["range24"][i]):
            continue
        trend = data["close"][i] > data["sma200"][i] and data["mom24"][i] > 0
        expansion = data["range"][i] >= 1.25 * data["range24"][i]
        breakout = (finite(data["prior_high48"][i])
                    and data["close"][i] > data["prior_high48"][i])
        if trend and expansion and (
                name == "trend_volatility" or breakout):
            eligible.append((data["mom24"][i], symbol))
    if not eligible:
        return None, 0
    return max(eligible)[1], 1


def summarize(
    rows: list[dict[str, float]],
    segment_start: int,
    segment_end: int,
) -> dict[str, Any]:
    block_values: list[list[float]] = [[] for _ in range(BLOCKS)]
    block_size = (segment_end - segment_start) / BLOCKS
    for row in rows:
        block = min(
            BLOCKS - 1,
            int((row["index"] - segment_start) / block_size),
        )
        block_values[block].append(row["gross_return"])

    counts = [len(values) for values in block_values]
    means = [
        sum(values) / len(values) if values else None
        for values in block_values
    ]
    valid = [value for value in means if value is not None]
    med = median(valid) if valid else math.nan
    stress_multiple = med / STRESS_COST if math.isfinite(med) else math.nan
    gate = (
        len(rows) >= BLOCKS * MIN_OBSERVATIONS_PER_BLOCK
        and all(count >= MIN_OBSERVATIONS_PER_BLOCK for count in counts)
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


def evaluate_segment(
    name: str,
    horizon: int,
    feat: dict[str, dict[str, list[float]]],
    pair_length: int,
    segment_start: int,
    segment_end: int,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    start = max(220, segment_start)
    for i in range(start, segment_end - horizon, horizon):
        symbol, direction = choose_signal(
            name, i, feat["BTC"], feat["ETH"]
        )
        if symbol is None:
            continue
        data = feat[symbol]
        gross = direction * (
            data["close"][i + horizon] / data["close"][i] - 1
        )
        rows.append({"index": i, "gross_return": gross})
    return summarize(rows, segment_start, segment_end)


def evaluate(btc_path: Path, eth_path: Path) -> dict[str, Any]:
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    required = 3 * 365 * 24
    if len(pair) < required:
        raise ValueError("need at least three years of aligned hourly candles")
    pair = pair[-required:]
    holdout_start = len(pair) - HOLDOUT_MIN_HOURS
    if holdout_start <= 220:
        raise ValueError("three-year window is too short for the frozen holdout")

    bars = {
        "BTC": [item.btc for item in pair],
        "ETH": [item.eth for item in pair],
    }
    feat = {symbol: features(values) for symbol, values in bars.items()}
    results: dict[str, Any] = {}

    for name in signal_names():
        results[name] = {}
        for horizon in HORIZONS:
            discovery = evaluate_segment(
                name, horizon, feat, len(pair), 0, holdout_start
            )
            holdout = evaluate_segment(
                name, horizon, feat, len(pair), holdout_start, len(pair)
            )
            results[name][str(horizon)] = {
                "candidate_id": f"{name}:{horizon}",
                "discovery": discovery,
                "holdout": holdout,
                "selected_for_holdout": discovery["passes_4x_stress_gate"],
                "passes_discovery_and_holdout": (
                    discovery["passes_4x_stress_gate"]
                    and holdout["passes_4x_stress_gate"]
                ),
            }

    passing = [
        f"{name}:{horizon}"
        for name, horizons in results.items()
        for horizon, result in horizons.items()
        if result["passes_discovery_and_holdout"]
    ]
    return {
        "schema_version": 2,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
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
            "completed_candles_only": True,
        },
        "costs": {
            "stress_round_trip": STRESS_COST,
            "stress_round_trip_bps": STRESS_EXECUTION.round_trip_bps,
        },
        "method": {
            "horizons_hours": HORIZONS,
            "sample_every_horizon": True,
            "minimum_observations_per_block": MIN_OBSERVATIONS_PER_BLOCK,
            "promotion_requires_discovery_and_holdout": True,
            "holdout_is_not_used_for_selection": True,
            "multiple_testing": {
                "family_size": FAMILY_SIZE,
                "correction": "chronological_holdout_confirmation",
                "every_candidate_horizon_requires_confirmation": True,
            },
        },
        "candidates": results,
        "passing_candidates": passing,
        "status": "candidate_found" if passing else "no_candidate_passed",
        "status_note": (
            "A passing gross diagnostic is not profitability evidence; "
            "fill-based net P&L is still required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.btc_path, args.eth_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "passing_candidates": report["passing_candidates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
