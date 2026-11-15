#!/usr/bin/env python3
"""Evaluate a frozen Binance open-interest expansion continuation hypothesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_momentum_volatility_research import Bar, load_bars
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import Bar, load_bars

START = datetime(2023, 10, 15, tzinfo=timezone.utc)
DISCOVERY_END = datetime(2024, 4, 15, tzinfo=timezone.utc)
END = datetime(2024, 10, 15, tzinfo=timezone.utc)
WARMUP_START = START - timedelta(hours=720)
PRICE_MOVE_THRESHOLD = 0.005
OI_LOOKBACK_HOURS = 6
OI_EXPANSION_THRESHOLD = 0.01
HOLD_HOURS = 8
COOLDOWN_HOURS = 8
NOTIONAL = 3_000.0
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_open_interest(path: Path) -> dict[datetime, float]:
    result: dict[datetime, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _utc(row["timestamp"])
            value = float(row["sum_open_interest_value"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("open-interest value must be finite and positive")
            result[timestamp] = value
    return result


def _signal(
    bars: list[Bar],
    open_interest: dict[datetime, float],
    index: int,
) -> tuple[int, float, float] | None:
    if index < OI_LOOKBACK_HOURS or index == 0:
        return None
    timestamp = bars[index].timestamp
    prior_timestamp = timestamp - timedelta(hours=OI_LOOKBACK_HOURS)
    previous_bar = bars[index - 1]
    if (
        previous_bar.timestamp != timestamp - timedelta(hours=1)
        or timestamp not in open_interest
        or prior_timestamp not in open_interest
    ):
        return None
    price_move = bars[index].close / previous_bar.close - 1.0
    oi_change = open_interest[timestamp] / open_interest[prior_timestamp] - 1.0
    if abs(price_move) < PRICE_MOVE_THRESHOLD or oi_change < OI_EXPANSION_THRESHOLD:
        return None
    return (1 if price_move > 0 else -1), price_move, oi_change


def _trade(bars: list[Bar], signal_index: int, side: int, symbol: str) -> dict[str, Any] | None:
    entry_index = signal_index + 1 + STRESS_EXECUTION.latency_bars
    exit_index = entry_index + HOLD_HOURS
    if exit_index >= len(bars):
        return None
    entry = float(bars[entry_index].open)
    exit_price = float(bars[exit_index].close)
    if not math.isfinite(entry) or not math.isfinite(exit_price) or entry <= 0:
        return None
    gross_return = side * (exit_price / entry - 1.0)
    filled_notional = (
        NOTIONAL
        * STRESS_EXECUTION.fill_fraction
        * (1.0 - STRESS_EXECUTION.outage_rejection_rate)
    )
    trading_cost = filled_notional * STRESS_EXECUTION.round_trip_bps / 10_000.0
    funding_cost = (
        filled_notional
        * STRESS_EXECUTION.funding_bps_per_bar
        * HOLD_HOURS
        / 10_000.0
    )
    net_pnl = filled_notional * gross_return - trading_cost - funding_cost
    return {
        "signal_timestamp": bars[signal_index].timestamp.isoformat(),
        "entry_timestamp": bars[entry_index].timestamp.isoformat(),
        "exit_timestamp": bars[exit_index].timestamp.isoformat(),
        "symbol": symbol,
        "side": "long" if side > 0 else "short",
        "gross_return": gross_return,
        "net_return": net_pnl / NOTIONAL,
        "net_pnl": net_pnl,
        "execution_cost": trading_cost + funding_cost,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = [[] for _ in range(BLOCKS)]
    for row in rows:
        groups[int(row["block_index"])].append(row)
    pnls = [float(row["net_pnl"]) for row in rows]
    returns = [float(row["net_return"]) for row in rows]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    equity = peak = max_drawdown = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    block_means = [
        statistics.mean(float(row["net_return"]) for row in group) if group else None
        for group in groups
    ]
    valid_means = [value for value in block_means if value is not None]
    mean_return = statistics.mean(returns) if returns else None
    volatility = statistics.pstdev(returns) if len(returns) > 1 else None
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_means,
        "net_pnl": sum(pnls),
        "net_return_pct": sum(returns) * 100.0,
        "max_drawdown_pct_of_notional": max_drawdown / NOTIONAL * 100.0,
        "sharpe_proxy": (
            mean_return / volatility * math.sqrt(len(returns))
            if mean_return is not None and volatility and volatility > 0
            else None
        ),
        "profit_factor": gains / losses if losses else None,
        "execution_cost": sum(float(row["execution_cost"]) for row in rows),
        "median_block_return_to_stress_cost": (
            statistics.median(valid_means)
            / (STRESS_EXECUTION.round_trip_bps / 10_000.0)
            if valid_means
            else None
        ),
        "passes_sample_gate": all(len(group) >= MIN_TRADES_PER_BLOCK for group in groups),
        "passes_positive_block_gate": (
            len(valid_means) == BLOCKS and all(value > 0 for value in valid_means)
        ),
    }


def _gate(summary: dict[str, Any]) -> bool:
    ratio = summary["median_block_return_to_stress_cost"]
    return bool(
        summary["passes_sample_gate"]
        and summary["passes_positive_block_gate"]
        and ratio is not None
        and ratio >= 1.0
    )


def _load_bars(path: Path) -> list[Bar]:
    bars = load_bars(path)
    if not bars or bars[0].timestamp > WARMUP_START or bars[-1].timestamp < END:
        raise ValueError(f"{path} does not cover the frozen window and warmup")
    return [bar for bar in bars if WARMUP_START <= bar.timestamp < END]


def _evaluate(
    bars: list[Bar],
    open_interest: dict[datetime, float],
    start: datetime,
    end: datetime,
    symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_index = next(i for i, bar in enumerate(bars) if bar.timestamp >= start)
    end_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars))
    rows: list[dict[str, Any]] = []
    last_signal_index = -10**9
    for index in range(start_index, max(start_index, end_index - HOLD_HOURS - 2)):
        signal = _signal(bars, open_interest, index)
        if signal is None:
            continue
        side, price_move, oi_change = signal
        if index - last_signal_index < COOLDOWN_HOURS:
            continue
        result = _trade(bars, index, side, symbol)
        if result is None:
            continue
        result.update({"price_move_1h": price_move, "oi_change_6h": oi_change})
        rows.append(result)
        last_signal_index = index
    rows.sort(key=lambda row: row["signal_timestamp"])
    start_epoch = int(start.timestamp() // 3600)
    width = (end - start).total_seconds() / 3600 / BLOCKS
    for row in rows:
        signal_epoch = _utc(row["signal_timestamp"]).timestamp() // 3600
        row["block_index"] = min(BLOCKS - 1, int((signal_epoch - start_epoch) / width))
    return rows, _summary(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--open-interest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bars_by_asset = {asset: _load_bars(args_path) for asset, args_path in {
        "BTC": args.btc_path, "ETH": args.eth_path
    }.items()}
    oi_by_asset = {
        asset: load_open_interest(args.open_interest_dir / f"{symbol}_1h.csv")
        for asset, symbol in ASSETS.items()
    }
    results: dict[str, Any] = {}
    for asset in ASSETS:
        discovery_rows, discovery = _evaluate(
            bars_by_asset[asset], oi_by_asset[asset], START, DISCOVERY_END, asset
        )
        holdout_rows, holdout = _evaluate(
            bars_by_asset[asset], oi_by_asset[asset], DISCOVERY_END, END, asset
        )
        results[asset] = {
            "discovery": discovery,
            "holdout": holdout,
            "passes_discovery": _gate(discovery),
            "passes_confirmation": bool(_gate(discovery) and _gate(holdout)),
            "status": "confirmed" if _gate(discovery) and _gate(holdout) else "not_confirmed",
            "discovery_trade_rows": discovery_rows,
            "holdout_trade_rows": holdout_rows,
        }

    manifests = {}
    for asset, symbol in ASSETS.items():
        manifest_path = args.open_interest_dir / f"{symbol}_1h.manifest.json"
        manifests[asset] = json.loads(manifest_path.read_text(encoding="utf-8"))

    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": (
            "A completed one-hour BTC/ETH price move with expanding USD-M "
            "open interest continues over the next eight hours."
        ),
        "frozen_parameters": {
            "signal_source": "Binance USD-M futures metrics",
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "price_move_threshold": PRICE_MOVE_THRESHOLD,
            "open_interest_measure": "sum_open_interest_value",
            "open_interest_lookback_hours": OI_LOOKBACK_HOURS,
            "open_interest_expansion_threshold": OI_EXPANSION_THRESHOLD,
            "direction": "price-up plus OI expansion -> long; price-down plus OI expansion -> short",
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "notional": NOTIONAL,
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": START.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "end_exclusive": END.isoformat(),
            "warmup_start": WARMUP_START.isoformat(),
            "evaluation_length_days": 366,
            "holdout_untouched": True,
            "completed_candles_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "newest_unseen_data_used": False,
            "window_is_independent_replication": False,
        },
        "source": {
            "provider": "Binance Vision",
            "market": "USD-M futures metrics",
            "archive_period": "5-minute metrics downsampled to the latest completed observation per hour",
            "manifests": manifests,
            "missing_data_is_excluded": True,
        },
        "candidates": results,
        "status": "confirmed" if any(value["passes_confirmation"] for value in results.values()) else "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        asset: {
            "status": value["status"],
            "discovery_trades": value["discovery"]["trade_count"],
            "holdout_trades": value["holdout"]["trade_count"],
        }
        for asset, value in results.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
