#!/usr/bin/env python3
"""Pre-registered edge-efficiency screen.

Measures directional forward movement after closed-candle signals and compares
it with explicit round-trip costs. Labels are sampled every holding horizon,
so overlapping windows are not counted as independent evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

BASE_COST = 0.0038
STRESS_COST = 0.0086
HORIZONS = (6, 12, 24, 48, 96)
BLOCKS = 6
MIN_OBSERVATIONS_PER_BLOCK = 20


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
    daily_range = [(h - l) / c if c else math.nan for h, l, c in zip(high, low, close)]
    mom24 = [math.nan if i < 24 else close[i] / close[i - 24] - 1 for i in range(len(close))]
    mom48 = [math.nan if i < 48 else close[i] / close[i - 48] - 1 for i in range(len(close))]
    sma200 = [rolling_mean(close, 200, i) for i in range(len(close))]
    range24 = [rolling_mean(daily_range, 24, i) for i in range(len(close))]
    prior_high48 = [max(high[i - 48:i]) if i >= 49 else math.nan for i in range(len(close))]
    return {"close": close, "mom24": mom24, "mom48": mom48,
            "sma200": sma200, "range": daily_range,
            "range24": range24, "prior_high48": prior_high48}


def signal_names() -> tuple[str, ...]:
    return ("trend_volatility", "breakout_volatility", "relative_strength_volatility")


def choose_signal(name: str, i: int, btc: dict[str, list[float]],
                  eth: dict[str, list[float]]) -> tuple[str | None, int]:
    if i < 220:
        return None, 0
    candidates = (("BTC", btc), ("ETH", eth))
    if name == "relative_strength_volatility":
        if not finite(btc["mom24"][i], eth["mom24"][i]):
            return None, 0
        leader, other = ("BTC", "ETH") if btc["mom24"][i] > eth["mom24"][i] else ("ETH", "BTC")
        lead, lag = (btc, eth) if leader == "BTC" else (eth, btc)
        if not finite(lead["range"][i], lead["range24"][i], lead["sma200"][i], lag["mom24"][i]):
            return None, 0
        if lead["mom24"][i] - lag["mom24"][i] >= 0.01 and lead["close"][i] > lead["sma200"][i] and lead["range"][i] >= 1.25 * lead["range24"][i]:
            return leader, 1
        return None, 0
    eligible = []
    for symbol, data in candidates:
        if not finite(data["mom24"][i], data["sma200"][i], data["range"][i], data["range24"][i]):
            continue
        trend = data["close"][i] > data["sma200"][i] and data["mom24"][i] > 0
        expansion = data["range"][i] >= 1.25 * data["range24"][i]
        breakout = finite(data["prior_high48"][i]) and data["close"][i] > data["prior_high48"][i]
        if trend and expansion and (name == "trend_volatility" or breakout):
            eligible.append((data["mom24"][i], symbol))
    return max(eligible)[1], 1 if eligible else (None, 0)


def evaluate(btc_path: Path, eth_path: Path) -> dict[str, Any]:
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    if len(pair) < 3 * 365 * 24:
        raise ValueError("need at least three years of aligned hourly candles")
    pair = pair[-3 * 365 * 24:]
    bars = {"BTC": [item.btc for item in pair], "ETH": [item.eth for item in pair]}
    feat = {symbol: features(values) for symbol, values in bars.items()}
    results: dict[str, Any] = {}
    for name in signal_names():
        results[name] = {}
        for horizon in HORIZONS:
            rows: list[dict[str, float]] = []
            start = 220
            for i in range(start, len(pair) - horizon, horizon):
                symbol, direction = choose_signal(name, i, feat["BTC"], feat["ETH"])
                if symbol is None:
                    continue
                data = feat[symbol]
                gross = direction * (data["close"][i + horizon] / data["close"][i] - 1)
                rows.append({"index": i, "gross_return": gross})
            block_values: list[list[float]] = [[] for _ in range(BLOCKS)]
            block_size = len(pair) / BLOCKS
            for row in rows:
                block = min(BLOCKS - 1, int(row["index"] / block_size))
                block_values[block].append(row["gross_return"])
            block_means = [sum(values) / len(values) if values else math.nan for values in block_values]
            valid_blocks = [value for value in block_means if math.isfinite(value)]
            base_multiple = median(valid_blocks) / BASE_COST if valid_blocks else math.nan
            stress_multiple = median(valid_blocks) / STRESS_COST if valid_blocks else math.nan
            eligible = (
                len(rows) >= MIN_OBSERVATIONS_PER_BLOCK * BLOCKS
                and len(valid_blocks) == BLOCKS
                and all(value > 0 for value in valid_blocks)
                and stress_multiple >= 4.0
            )
            results[name][str(horizon)] = {
                "observations": len(rows),
                "non_overlapping": True,
                "block_mean_gross_returns": block_means,
                "median_block_gross_return": median(valid_blocks) if valid_blocks else None,
                "base_cost_multiple": base_multiple if math.isfinite(base_multiple) else None,
                "stress_cost_multiple": stress_multiple if math.isfinite(stress_multiple) else None,
                "passes_4x_stress_gate": eligible,
            }
    passing = [f"{name}:{horizon}" for name, horizons in results.items()
               for horizon, result in horizons.items() if result["passes_4x_stress_gate"]]
    return {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "window": {"start": pair[0].timestamp.isoformat(), "end": pair[-1].timestamp.isoformat(),
                   "bars": len(pair), "blocks": BLOCKS, "completed_candles_only": True},
        "costs": {"base_round_trip": BASE_COST, "stress_round_trip": STRESS_COST},
        "method": {"horizons_hours": HORIZONS, "sample_every_horizon": True,
                   "minimum_observations_per_block": MIN_OBSERVATIONS_PER_BLOCK,
                   "promotion_requires_4x_stress_and_positive_blocks": True},
        "candidates": results,
        "passing_candidates": passing,
        "status": "candidate_found" if passing else "no_candidate_passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.btc_path, args.eth_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "passing_candidates": report["passing_candidates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
