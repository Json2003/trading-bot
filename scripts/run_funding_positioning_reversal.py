#!/usr/bin/env python3
"""Frozen funding-positioning reversal experiment; research-only."""
from __future__ import annotations

import argparse
import csv
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
FUNDING_THRESHOLD = 0.0008
HOLD_HOURS = 8
COOLDOWN_HOURS = 8
NOTIONAL = 3_000.0
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def load_funding(path: Path) -> dict[datetime, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            datetime.fromisoformat(row["timestamp"]): float(row["funding_rate"])
            for row in csv.DictReader(handle)
        }


def summary(rows: list[dict], start: int, end: int) -> dict:
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
    equity = peak = max_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    sharpe = None
    if len(returns) > 1 and pstdev(returns) > 0:
        sharpe = mean(returns) / pstdev(returns) * math.sqrt(365 * 24 / HOLD_HOURS)
    valid = [value for value in block_means if value is not None]
    median_block = median(valid) if valid else None
    stress_cost = STRESS_EXECUTION.round_trip_bps / 10_000.0
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
            median_block / stress_cost if median_block is not None else None
        ),
        "passes_sample_gate": all(len(group) >= MIN_TRADES_PER_BLOCK for group in groups),
        "passes_positive_block_gate": (
            len(valid) == BLOCKS and all(value > 0 for value in valid)
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
    funding_cost = filled * STRESS_EXECUTION.funding_bps_per_bar * HOLD_HOURS / 10_000.0
    net_pnl = filled * gross - trading_cost - funding_cost
    return {
        "index": index,
        "symbol": symbol,
        "side": "long" if side > 0 else "short",
        "funding_rate": None,
        "gross_return": gross,
        "net_return": net_pnl / NOTIONAL,
        "net_pnl": net_pnl,
        "execution_cost": trading_cost + funding_cost,
    }


def evaluate(
    pair: list,
    funding: dict[str, dict[datetime, float]],
    start: int,
    end: int,
) -> list[dict]:
    index_by_time = {item.btc.timestamp: index for index, item in enumerate(pair)}
    events = []
    for symbol, rates in funding.items():
        for timestamp, rate in rates.items():
            if timestamp in index_by_time and START <= timestamp < END:
                events.append((index_by_time[timestamp], symbol, rate))
    events.sort()
    last_signal = {"BTC": -10**9, "ETH": -10**9}
    rows = []
    for index, symbol, rate in events:
        if index < start or index >= end or index - last_signal[symbol] < COOLDOWN_HOURS:
            continue
        side = -1 if rate >= FUNDING_THRESHOLD else 1 if rate <= -FUNDING_THRESHOLD else 0
        if side == 0:
            continue
        row = trade(pair, symbol, side, index)
        if row is not None:
            row["funding_rate"] = rate
            rows.append(row)
            last_signal[symbol] = index
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--btc-funding-path", type=Path, required=True)
    parser.add_argument("--eth-funding-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aligned = align_pair(load_bars(args.btc_path), load_bars(args.eth_path))
    pair = [item for item in aligned if START <= item.btc.timestamp < END]
    if not pair or pair[-1].btc.timestamp < DISCOVERY_END:
        raise ValueError("fixed evaluation window is incomplete")
    discovery_end = next(
        index for index, item in enumerate(pair) if item.btc.timestamp >= DISCOVERY_END
    )
    funding = {
        "BTC": load_funding(args.btc_funding_path),
        "ETH": load_funding(args.eth_funding_path),
    }
    discovery_rows = evaluate(pair, funding, 0, discovery_end)
    holdout_rows = evaluate(pair, funding, discovery_end, len(pair))
    discovery = summary(discovery_rows, 0, discovery_end)
    holdout = summary(holdout_rows, discovery_end, len(pair))
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
        "hypothesis": "extreme funding indicates crowded positioning that reverses over the next eight hours",
        "frozen_parameters": {
            "assets": ["BTCUSDT", "ETHUSDT"],
            "funding_threshold": FUNDING_THRESHOLD,
            "funding_threshold_bps": FUNDING_THRESHOLD * 10_000,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "direction": "positive funding short; negative funding long",
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
        "funding_data": {
            "source": "Binance USD-M monthly funding-rate archives",
            "btc_rows": len(funding["BTC"]),
            "eth_rows": len(funding["ETH"]),
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
