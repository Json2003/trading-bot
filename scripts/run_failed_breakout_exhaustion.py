#!/usr/bin/env python3
"""Frozen, research-only failed-breakout exhaustion hypothesis.

A completed hourly candle qualifies when volume and true range are both
unusually large, the candle breaches one side of the prior 24-hour range, and
the close returns inside that range. An upside rejection is shorted and a
downside rejection is bought for four hours with one-bar latency. Ambiguous
candles that breach both sides are excluded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

LOOKBACK_HOURS = 24
WARMUP_HOURS = 220
HOLD_BARS = 4
LATENCY_BARS = 1  # Signal candle to entry candle: next candle's open.
REQUIRED_HOURS = 3 * 365 * 24
HOLDOUT_HOURS = 11_520
BLOCKS = 6
MIN_PER_BLOCK = 20
NOTIONAL = 6_000.0
VOLUME_MULTIPLE = 2.0
RANGE_MULTIPLE = 1.5
FILL_FRACTION = STRESS_EXECUTION.fill_fraction
REJECTION_RATE = STRESS_EXECUTION.outage_rejection_rate
ROUND_TRIP_BPS = STRESS_EXECUTION.round_trip_bps
FUNDING_BPS_PER_BAR = STRESS_EXECUTION.funding_bps_per_bar
STRESS_COST = ROUND_TRIP_BPS / 10_000.0


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def prior_median(values: list[float], index: int, period: int) -> float:
    if index < period:
        return math.nan
    return float(median(values[index - period:index]))


def features(bars):
    close = [float(bar.close) for bar in bars]
    high = [float(bar.high) for bar in bars]
    low = [float(bar.low) for bar in bars]
    volume = [float(bar.volume) for bar in bars]
    true_range = []
    for index, (h, l, c) in enumerate(zip(high, low, close)):
        if c <= 0:
            true_range.append(math.nan)
            continue
        previous = close[index - 1] if index else c
        true_range.append(max(h - l, abs(h - previous), abs(l - previous)) / c)
    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "true_range": true_range,
        "volume_baseline": [
            prior_median(volume, i, LOOKBACK_HOURS)
            for i in range(len(close))
        ],
        "range_baseline": [
            prior_median(true_range, i, LOOKBACK_HOURS)
            for i in range(len(close))
        ],
        "prior_high": [
            max(high[i - LOOKBACK_HOURS:i])
            if i >= LOOKBACK_HOURS else math.nan
            for i in range(len(close))
        ],
        "prior_low": [
            min(low[i - LOOKBACK_HOURS:i])
            if i >= LOOKBACK_HOURS else math.nan
            for i in range(len(close))
        ],
    }


def candidate(index: int, btc: dict, eth: dict):
    choices = []
    for symbol, data in (("BTC", btc), ("ETH", eth)):
        values = (
            data["close"][index],
            data["high"][index],
            data["low"][index],
            data["volume"][index],
            data["true_range"][index],
            data["volume_baseline"][index],
            data["range_baseline"][index],
            data["prior_high"][index],
            data["prior_low"][index],
        )
        if not finite(*values):
            continue
        volume_ratio = data["volume"][index] / data["volume_baseline"][index]
        range_ratio = data["true_range"][index] / data["range_baseline"][index]
        if volume_ratio < VOLUME_MULTIPLE or range_ratio < RANGE_MULTIPLE:
            continue

        upside_rejection = (
            data["high"][index] > data["prior_high"][index]
            and data["close"][index] <= data["prior_high"][index]
        )
        downside_rejection = (
            data["low"][index] < data["prior_low"][index]
            and data["close"][index] >= data["prior_low"][index]
        )
        if upside_rejection == downside_rejection:
            continue

        direction = -1 if upside_rejection else 1
        choices.append({
            "symbol": symbol,
            "direction": direction,
            "rejection": "upside" if direction < 0 else "downside",
            "volume_ratio": volume_ratio,
            "range_ratio": range_ratio,
            "score": volume_ratio * range_ratio,
        })
    if not choices:
        return None
    return max(choices, key=lambda row: (row["score"], row["symbol"]))


def trade_return(pair, symbol: str, direction: int, signal_index: int):
    entry_index = signal_index + LATENCY_BARS
    # Four hourly candles from entry open through the fourth candle's close.
    exit_index = entry_index + HOLD_BARS - 1
    if exit_index >= len(pair):
        return None
    entry_bar = pair[entry_index].btc if symbol == "BTC" else pair[entry_index].eth
    exit_bar = pair[exit_index].btc if symbol == "BTC" else pair[exit_index].eth
    entry = float(entry_bar.open)
    exit_price = float(exit_bar.close)
    if entry <= 0 or exit_price <= 0:
        return None
    gross = direction * (exit_price / entry - 1.0)
    filled_notional = NOTIONAL * FILL_FRACTION * (1.0 - REJECTION_RATE)
    trading_cost = filled_notional * ROUND_TRIP_BPS / 10_000.0
    funding_cost = (
        filled_notional * FUNDING_BPS_PER_BAR * HOLD_BARS / 10_000.0
    )
    net_pnl = filled_notional * gross - trading_cost - funding_cost
    return {
        "signal_index": signal_index,
        "symbol": symbol,
        "direction": "LONG" if direction == 1 else "SHORT",
        "gross_return": gross,
        "net_return": net_pnl / NOTIONAL,
        "net_pnl": net_pnl,
        "execution_cost": trading_cost + funding_cost,
    }


def collect_segment(pair, btc, eth, start: int, end: int):
    rows = []
    index = max(start, WARMUP_HOURS)
    # end is exclusive: every exit must remain inside this segment.
    last_allowed = end - (LATENCY_BARS + HOLD_BARS)
    while index <= last_allowed:
        selected = candidate(index, btc, eth)
        if selected is None:
            index += 1
            continue
        row = trade_return(pair, selected["symbol"], selected["direction"], index)
        if row is not None:
            row.update({
                "rejection": selected["rejection"],
                "volume_ratio": selected["volume_ratio"],
                "range_ratio": selected["range_ratio"],
            })
            rows.append(row)
        index += LATENCY_BARS + HOLD_BARS
    return rows


def summarize(rows, start: int, end: int) -> dict:
    groups = [[] for _ in range(BLOCKS)]
    width = (end - start) / BLOCKS
    for row in rows:
        block = min(
            BLOCKS - 1,
            max(0, int((row["signal_index"] - start) / width)),
        )
        groups[block].append(row)

    values = [float(row["net_return"]) for row in rows]
    pnl = [float(row["net_pnl"]) for row in rows]
    block_returns = [
        mean([float(row["net_return"]) for row in group])
        if group else None
        for group in groups
    ]
    gains = sum(value for value in pnl if value > 0)
    losses = abs(sum(value for value in pnl if value < 0))
    equity = peak = drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)

    sharpe = None
    if len(values) > 1 and pstdev(values) > 0:
        sharpe = (
            mean(values) / pstdev(values)
            * math.sqrt(365 * 24 / HOLD_BARS)
        )
    valid_blocks = [value for value in block_returns if value is not None]
    median_block = median(valid_blocks) if valid_blocks else None
    passes = bool(
        len(rows) >= BLOCKS * MIN_PER_BLOCK
        and all(len(group) >= MIN_PER_BLOCK for group in groups)
        and len(valid_blocks) == BLOCKS
        and all(value > 0 for value in valid_blocks)
        and median_block is not None
        and median_block >= 4.0 * STRESS_COST
    )
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_returns,
        "net_return_pct_of_fixed_notional": sum(values) * 100.0,
        "max_drawdown_pct_of_fixed_notional": drawdown / NOTIONAL * 100.0,
        "sharpe_annualized_trade_proxy": sharpe,
        "profit_factor": gains / losses if losses else None,
        "win_rate": (
            sum(1 for value in pnl if value > 0) / len(pnl)
            if pnl else None
        ),
        "execution_costs": sum(
            float(row["execution_cost"]) for row in rows
        ),
        "median_block_net_return": median_block,
        "median_block_to_stress_cost": (
            median_block / STRESS_COST if median_block is not None else None
        ),
        "all_blocks_minimum_sample": all(
            len(group) >= MIN_PER_BLOCK for group in groups
        ),
        "all_blocks_positive": bool(
            len(valid_blocks) == BLOCKS
            and all(value > 0 for value in valid_blocks)
        ),
        "passes_frozen_gate": passes,
    }


def run(btc_path: Path, eth_path: Path) -> dict:
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    if len(pair) < REQUIRED_HOURS:
        raise ValueError("three years of aligned hourly candles are required")
    pair = pair[-REQUIRED_HOURS:]
    holdout_start = len(pair) - HOLDOUT_HOURS
    if holdout_start <= WARMUP_HOURS:
        raise ValueError("holdout leaves insufficient discovery history")

    btc = features([row.btc for row in pair])
    eth = features([row.eth for row in pair])
    discovery = collect_segment(pair, btc, eth, WARMUP_HOURS, holdout_start)
    holdout = collect_segment(pair, btc, eth, holdout_start, len(pair))
    discovery_summary = summarize(discovery, WARMUP_HOURS, holdout_start)
    holdout_summary = summarize(holdout, holdout_start, len(pair))
    confirmed = (
        discovery_summary["passes_frozen_gate"]
        and holdout_summary["passes_frozen_gate"]
    )
    return {
        "schema_version": 1,
        "research_only": True,
        "orders_placed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
        "risk_limits_changed": False,
        "promotion_allowed": False,
        "hypothesis": {
            "name": "failed_breakout_exhaustion_reversal",
            "description": (
                "Completed hourly volume >= 2x prior 24h median and true "
                "range >= 1.5x prior 24h median; an upside breach that "
                "closes back inside the range is shorted, while a downside "
                "breach that closes back inside is bought."
            ),
            "volume_multiple": VOLUME_MULTIPLE,
            "range_multiple": RANGE_MULTIPLE,
            "hold_bars": HOLD_BARS,
            "latency_bars": LATENCY_BARS,
            "entry_timing": "open of signal_index + latency_bars",
            "exit_timing": "close of entry_index + hold_bars - 1",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "ambiguous_two_sided_breaches": "excluded",
        },
        "window": {
            "start": pair[0].timestamp.isoformat(),
            "end": pair[-1].timestamp.isoformat(),
            "bars": len(pair),
            "discovery_bars": holdout_start,
            "holdout_start_index": holdout_start,
            "holdout_bars": HOLDOUT_HOURS,
            "completed_candles_only": True,
            "portfolio_non_overlapping_windows": True,
        },
        "data_freshness": {
            "last_completed_candle": pair[-1].timestamp.isoformat(),
            "source": "Binance Vision normalized hourly klines",
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "segments": {
            "discovery": discovery_summary,
            "untouched_holdout": holdout_summary,
        },
        "confirmed_candidate": confirmed,
        "status": (
            "candidate_advances_toward_confirmation"
            if confirmed else "no_confirmation"
        ),
        "status_note": (
            "Both discovery and untouched holdout must pass every frozen "
            "gate; no promotion is automatic."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.btc_path, args.eth_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "last_completed_candle": report["data_freshness"]["last_completed_candle"],
        "discovery": report["segments"]["discovery"],
        "untouched_holdout": report["segments"]["untouched_holdout"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
