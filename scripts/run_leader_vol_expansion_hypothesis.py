#!/usr/bin/env python3
"""Paper-only leader/volatility-expansion hypothesis test.

The candidate is intentionally pre-registered: long-only, BTC/ETH leader
selection, two-bar confirmation, volatility expansion, next-bar entry with
one-bar latency, 24/48/96-hour non-overlapping holds, shared stress costs,
and a frozen holdout. It never places orders or enables leverage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

try:
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

HORIZONS = (24, 48, 96)
HOURS_PER_YEAR = 24 * 365
HOLDOUT_HOURS = 11_520
BLOCKS = 6
MIN_PER_BLOCK = 20
NOTIONAL = 6_000.0
FEE_BPS = 20.0
SPREAD_BPS = 10.0
SLIPPAGE_BPS = 10.0
IMPACT_BPS = 8.0
LATENCY_BARS = 1
FILL_FRACTION = 0.80
FUNDING_BPS_PER_BAR = 0.5
REJECTION_RATE = 0.02
EFFECTIVE_SLIP_BPS = SPREAD_BPS / 2 + SLIPPAGE_BPS + IMPACT_BPS
ROUND_TRIP_BPS = 2 * (FEE_BPS + EFFECTIVE_SLIP_BPS)


def finite(*values):
    return all(math.isfinite(float(v)) for v in values)


def ema(values, period):
    if len(values) < period:
        return [math.nan] * len(values)
    out = [math.nan] * (period - 1)
    value = mean(values[:period])
    out.append(value)
    alpha = 2 / (period + 1)
    for value_i in values[period:]:
        value = alpha * value_i + (1 - alpha) * value
        out.append(value)
    return out


def rolling_mean(values, period):
    out = [math.nan] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = mean(values[i - period + 1:i + 1])
    return out


def features(bars):
    close = [float(x.close) for x in bars]
    high = [float(x.high) for x in bars]
    low = [float(x.low) for x in bars]
    ranges = [(h - l) / c if c else math.nan for h, l, c in zip(high, low, close)]
    return {
        "close": close,
        "ema200": ema(close, 200),
        "range": ranges,
        "avg_range24": rolling_mean(ranges, 24),
        "ret24": [math.nan if i < 24 else close[i] / close[i - 24] - 1 for i in range(len(close))],
    }


def choose(i, btc, eth):
    if i < 220:
        return None
    candidates = []
    for symbol, x, other in (("BTC", btc, eth), ("ETH", eth, btc)):
        if not finite(x["close"][i], x["ema200"][i], x["ret24"][i],
                      x["avg_range24"][i], x["range"][i],
                      x["close"][i - 1], x["close"][i - 2]):
            continue
        two_bar = x["close"][i - 1] > x["close"][i - 2] and x["close"][i] > x["close"][i - 1]
        leader_gap = x["ret24"][i] - other["ret24"][i]
        expansion = x["range"][i] >= 1.25 * x["avg_range24"][i]
        if x["close"][i] > x["ema200"][i] and x["ret24"][i] > 0 and leader_gap >= 0.01 and two_bar and expansion:
            candidates.append((leader_gap, symbol))
    return max(candidates)[1] if candidates else None


def trade_return(pair, symbol, i, horizon, data):
    entry_i = i + 1 + LATENCY_BARS
    exit_i = entry_i + horizon
    if exit_i >= len(pair):
        return None
    bar = pair[entry_i].btc if symbol == "BTC" else pair[entry_i].eth
    exit_bar = pair[exit_i].btc if symbol == "BTC" else pair[exit_i].eth
    entry = float(bar.open)
    exit_price = float(exit_bar.close)
    if entry <= 0:
        return None
    gross = exit_price / entry - 1
    filled_notional = NOTIONAL * FILL_FRACTION * (1 - REJECTION_RATE)
    trading_cost = filled_notional * ROUND_TRIP_BPS / 10_000
    funding_cost = filled_notional * FUNDING_BPS_PER_BAR * horizon / 10_000
    net_pnl = filled_notional * gross - trading_cost - funding_cost
    net_return = net_pnl / NOTIONAL
    return {
        "index": i,
        "symbol": symbol,
        "gross_return": gross,
        "net_return": net_return,
        "net_pnl": net_pnl,
        "execution_cost": trading_cost + funding_cost,
    }


def summarize(rows, start, end):
    groups = [[] for _ in range(BLOCKS)]
    width = (end - start) / BLOCKS
    for row in rows:
        block = min(BLOCKS - 1, int((row["index"] - start) / width))
        groups[block].append(row)
    counts = [len(g) for g in groups]
    block_returns = [mean([r["net_return"] for r in g]) if g else None for g in groups]
    values = [r["net_return"] for r in rows]
    gross = [r["gross_return"] for r in rows]
    pnl = [r["net_pnl"] for r in rows]
    costs = [r["execution_cost"] for r in rows]
    gains = sum(x for x in pnl if x > 0)
    losses = abs(sum(x for x in pnl if x < 0))
    sharpe = None
    if len(values) > 1 and pstdev(values) > 0:
        sharpe = mean(values) / pstdev(values) * math.sqrt(max(1, HOURS_PER_YEAR / 24))
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    stress_cost = ROUND_TRIP_BPS / 10_000
    med_gross = median(gross) if gross else None
    return {
        "observations": len(rows),
        "block_observations": counts,
        "block_mean_net_returns": block_returns,
        "net_return_pct": sum(values) * 100 if values else 0.0,
        "max_drawdown_pct_of_notional": drawdown / NOTIONAL * 100,
        "sharpe_annualized_trade_proxy": sharpe,
        "profit_factor": gains / losses if losses else None,
        "trade_count": len(rows),
        "execution_cost": sum(costs),
        "median_gross_return": med_gross,
        "median_gross_to_stress_cost": med_gross / stress_cost if med_gross is not None else None,
        "all_blocks_minimum_sample": all(c >= MIN_PER_BLOCK for c in counts),
        "all_blocks_positive": all(v is not None and v > 0 for v in block_returns),
    }


def run(btc_path, eth_path):
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    required = 3 * 365 * 24
    if len(pair) < required:
        raise ValueError("three years of aligned hourly candles are required")
    pair = pair[-required:]
    btc = features([x.btc for x in pair])
    eth = features([x.eth for x in pair])
    holdout_start = len(pair) - HOLDOUT_HOURS
    results = {}
    for horizon in HORIZONS:
        discovery, holdout = [], []
        for i in range(220, holdout_start - horizon, horizon):
            symbol = choose(i, btc, eth)
            if symbol:
                row = trade_return(pair, symbol, i, horizon, {"BTC": btc, "ETH": eth})
                if row:
                    discovery.append(row)
        for i in range(holdout_start, len(pair) - horizon, horizon):
            symbol = choose(i, btc, eth)
            if symbol:
                row = trade_return(pair, symbol, i, horizon, {"BTC": btc, "ETH": eth})
                if row:
                    holdout.append(row)
        d = summarize(discovery, 220, holdout_start)
        h = summarize(holdout, holdout_start, len(pair) - horizon)
        results[str(horizon)] = {
            "candidate_id": f"leader_vol_expansion_{horizon}h",
            "discovery": d,
            "holdout": h,
            "passes_discovery": d["all_blocks_minimum_sample"] and d["all_blocks_positive"],
            "passes_holdout": h["all_blocks_minimum_sample"] and h["all_blocks_positive"],
            "passes_confirmation": (
                d["all_blocks_minimum_sample"] and d["all_blocks_positive"]
                and h["all_blocks_minimum_sample"] and h["all_blocks_positive"]
            ),
        }
    return {
        "schema_version": 1,
        "research_only": True,
        "orders_placed": False,
        "leverage_enabled": False,
        "window": {
            "start": pair[0].timestamp.isoformat(),
            "end": pair[-1].timestamp.isoformat(),
            "bars": len(pair),
            "holdout_hours": HOLDOUT_HOURS,
            "completed_candles_only": True,
            "non_overlapping": True,
        },
        "execution_model": {
            "fee_bps_per_side": FEE_BPS,
            "spread_bps_per_side": SPREAD_BPS,
            "slippage_bps_per_side": SLIPPAGE_BPS,
            "impact_bps_per_side": IMPACT_BPS,
            "round_trip_bps": ROUND_TRIP_BPS,
            "latency_bars": LATENCY_BARS,
            "fill_fraction": FILL_FRACTION,
            "funding_bps_per_bar": FUNDING_BPS_PER_BAR,
            "rejection_rate": REJECTION_RATE,
        },
        "hypothesis": "leader selection plus volatility expansion plus two-bar confirmation",
        "horizons": results,
        "confirmed_candidates": [k for k, v in results.items() if v["passes_confirmation"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.btc_path, args.eth_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"confirmed_candidates": report["confirmed_candidates"], "horizons": list(report["horizons"])}, indent=2))


if __name__ == "__main__":
    main()
