#!/usr/bin/env python3
"""Frozen directional-magnitude baseline; research-only, no orders."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
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

START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 1, tzinfo=timezone.utc)
DISCOVERY_END = datetime(2025, 4, 1, tzinfo=timezone.utc)
LOOKBACK_HOURS = 3
HOLD_HOURS = 6
COOLDOWN_HOURS = 6
NOTIONAL = 3_000.0
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20
MOVE_THRESHOLD = STRESS_EXECUTION.round_trip_bps / 10_000.0


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def summarize(rows: list[dict], start: int, end: int) -> dict:
    width = (end - start) / BLOCKS
    groups = [[] for _ in range(BLOCKS)]
    for row in rows:
        block = min(BLOCKS - 1, int((row["index"] - start) / width))
        groups[block].append(row)
    pnl = [row["net_pnl"] for row in rows]
    returns = [row["net_return"] for row in rows]
    block_means = [
        mean([row["net_return"] for row in group]) if group else None
        for group in groups
    ]
    gains = sum(value for value in pnl if value > 0)
    losses = abs(sum(value for value in pnl if value < 0))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    sharpe = None
    if len(returns) > 1 and pstdev(returns) > 0:
        sharpe = mean(returns) / pstdev(returns) * math.sqrt(
            365 * 24 / HOLD_HOURS
        )
    valid_blocks = [value for value in block_means if value is not None]
    median_block = median(valid_blocks) if valid_blocks else None
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_means,
        "net_pnl": sum(pnl),
        "net_return_pct": sum(returns) * 100.0,
        "max_drawdown_pct_of_notional": max_drawdown / NOTIONAL * 100.0,
        "sharpe_annualized_trade_proxy": sharpe,
        "profit_factor": gains / losses if losses else None,
        "execution_cost": sum(row["execution_cost"] for row in rows),
        "median_block_return_to_stress_cost": (
            median_block / MOVE_THRESHOLD if median_block is not None else None
        ),
        "passes_sample_gate": all(
            len(group) >= MIN_TRADES_PER_BLOCK for group in groups
        ),
        "passes_positive_block_gate": (
            len(valid_blocks) == BLOCKS and all(value > 0 for value in valid_blocks)
        ),
    }


def trade(pair: list, symbol: str, side: int, index: int) -> dict | None:
    entry_index = index + 1 + STRESS_EXECUTION.latency_bars
    exit_index = entry_index + HOLD_HOURS
    if exit_index >= len(pair):
        return None
    entry_bar = pair[entry_index].btc if symbol == "BTC" else pair[entry_index].eth
    exit_bar = pair[exit_index].btc if symbol == "BTC" else pair[exit_index].eth
    entry = float(entry_bar.open)
    exit_price = float(exit_bar.close)
    if entry <= 0:
        return None
    gross = side * (exit_price / entry - 1.0)
    filled = NOTIONAL * STRESS_EXECUTION.fill_fraction * (
        1.0 - STRESS_EXECUTION.outage_rejection_rate
    )
    trading_cost = filled * STRESS_EXECUTION.round_trip_bps / 10_000.0
    funding_cost = (
        filled * STRESS_EXECUTION.funding_bps_per_bar * HOLD_HOURS / 10_000.0
    )
    net_pnl = filled * gross - trading_cost - funding_cost
    return {
        "index": index,
        "symbol": symbol,
        "side": "long" if side == 1 else "short",
        "gross_return": gross,
        "net_return": net_pnl / NOTIONAL,
        "net_pnl": net_pnl,
        "execution_cost": trading_cost + funding_cost,
    }


def evaluate(pair: list, start: int, end: int) -> list[dict]:
    closes = {
        "BTC": [float(item.btc.close) for item in pair],
        "ETH": [float(item.eth.close) for item in pair],
    }
    last_signal = {"BTC": -10**9, "ETH": -10**9}
    rows = []
    for index in range(max(start, LOOKBACK_HOURS), end - HOLD_HOURS):
        for symbol in ("BTC", "ETH"):
            if index - last_signal[symbol] < COOLDOWN_HOURS:
                continue
            previous = closes[symbol][index - LOOKBACK_HOURS]
            current = closes[symbol][index]
            if previous <= 0:
                continue
            move = current / previous - 1.0
            if not finite(move) or abs(move) < MOVE_THRESHOLD:
                continue
            side = 1 if move > 0 else -1
            row = trade(pair, symbol, side, index)
            if row is not None:
                rows.append(row)
                last_signal[symbol] = index
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aligned = align_pair(load_bars(args.btc_path), load_bars(args.eth_path))
    pair = [
        item for item in aligned
        if START <= item.btc.timestamp < END
    ]
    if not pair or pair[-1].btc.timestamp < DISCOVERY_END:
        raise ValueError("fixed evaluation window is incomplete")
    discovery_end = next(
        index for index, item in enumerate(pair)
        if item.btc.timestamp >= DISCOVERY_END
    )
    discovery_rows = evaluate(pair, 0, discovery_end)
    holdout_rows = evaluate(pair, discovery_end, len(pair))
    discovery = summarize(discovery_rows, 0, discovery_end)
    holdout = summarize(holdout_rows, discovery_end, len(pair))
    passes_discovery = (
        discovery["passes_sample_gate"]
        and discovery["passes_positive_block_gate"]
        and (discovery["median_block_return_to_stress_cost"] or 0.0) >= 1.0
    )
    passes_confirmation = (
        passes_discovery
        and holdout["passes_sample_gate"]
        and holdout["passes_positive_block_gate"]
        and (holdout["median_block_return_to_stress_cost"] or 0.0) >= 1.0
    )
    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "hypothesis": "directional continuation after a fixed-magnitude three-hour move",
        "frozen_parameters": {
            "assets": ["BTCUSDT", "ETHUSDT"],
            "lookback_hours": LOOKBACK_HOURS,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "movement_threshold": MOVE_THRESHOLD,
            "movement_threshold_bps": STRESS_EXECUTION.round_trip_bps,
            "direction": "long after positive move; short after negative move",
            "position_selection": "both assets independently; no leader selection",
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": pair[0].btc.timestamp.isoformat(),
            "end": pair[-1].btc.timestamp.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "holdout_untouched": True,
            "completed_candles_only": True,
            "non_overlapping_per_asset": True,
        },
        "discovery": discovery,
        "holdout": holdout,
        "passes_discovery": passes_discovery,
        "passes_confirmation": passes_confirmation,
        "status": "confirmed" if passes_confirmation else "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "discovery_trades": discovery["trade_count"],
        "holdout_trades": holdout["trade_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
