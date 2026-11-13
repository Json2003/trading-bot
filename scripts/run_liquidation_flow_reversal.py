#!/usr/bin/env python3
"""Evaluate one frozen liquidation-flow reversal hypothesis.

This is a research-only diagnostic. It never places orders, enables leverage,
changes the active profile, or uses the confirmation slice for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_momentum_volatility_research import Bar, load_bars
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import Bar, load_bars

START = datetime(2023, 6, 25, tzinfo=timezone.utc)
DISCOVERY_END = datetime(2024, 2, 25, tzinfo=timezone.utc)
END = datetime(2024, 10, 15, tzinfo=timezone.utc)
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
LIQUIDATION_FILES = {
    "BTC": "BTCUSD_PERP_liquidations.csv",
    "ETH": "ETHUSD_PERP_liquidations.csv",
}
BASELINE_HOURS = 720
MIN_BASELINE_EVENTS = 30
EXTREME_MULTIPLIER = 3.0
DOMINANCE_THRESHOLD = 0.60
HOLD_HOURS = 8
COOLDOWN_HOURS = 8
NOTIONAL = 3_000.0
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_liquidations(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            side = row["side"].upper()
            if side not in {"BUY", "SELL"}:
                raise ValueError(f"unexpected liquidation side: {side}")
            amount = float(row["liquidation_usd"])
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("liquidation_usd must be finite and positive")
            timestamp = _utc(row["timestamp"]).replace(minute=0, second=0, microsecond=0)
            rows.append({"timestamp": timestamp, "side": side, "liquidation_usd": amount})
    return rows


def aggregate_hourly(rows: list[dict[str, Any]]) -> dict[datetime, dict[str, float]]:
    result: dict[datetime, dict[str, float]] = {}
    for row in rows:
        timestamp = row["timestamp"].replace(minute=0, second=0, microsecond=0)
        bucket = result.setdefault(
            timestamp,
            {"buy_usd": 0.0, "sell_usd": 0.0},
        )
        bucket["buy_usd" if row["side"] == "BUY" else "sell_usd"] += row["liquidation_usd"]
    for bucket in result.values():
        bucket["total_usd"] = bucket["buy_usd"] + bucket["sell_usd"]
    return result


def _signal(
    flow: dict[datetime, dict[str, float]],
    timestamps: list[datetime],
    index: int,
    missing_dates: set[date] | None = None,
) -> int:
    missing_dates = missing_dates or set()
    prior_timestamps = timestamps[max(0, index - BASELINE_HOURS):index]
    if timestamps[index].date() in missing_dates or any(
        timestamp.date() in missing_dates for timestamp in prior_timestamps
    ):
        return 0
    current = flow.get(
        timestamps[index],
        {"buy_usd": 0.0, "sell_usd": 0.0, "total_usd": 0.0},
    )
    prior = [
        flow.get(timestamp, {"total_usd": 0.0})["total_usd"]
        for timestamp in timestamps[max(0, index - BASELINE_HOURS):index]
    ]
    nonzero = [value for value in prior if value > 0]
    if len(nonzero) < MIN_BASELINE_EVENTS or not current["total_usd"]:
        return 0
    baseline = statistics.median(nonzero)
    if current["total_usd"] < EXTREME_MULTIPLIER * baseline:
        return 0
    dominant = max(current["buy_usd"], current["sell_usd"]) / current["total_usd"]
    if dominant < DOMINANCE_THRESHOLD:
        return 0
    # BUY liquidations close shorts; SELL liquidations close longs. The
    # reversal trades against the forced-flow direction after the flush.
    return 1 if current["sell_usd"] > current["buy_usd"] else -1


def trade(
    bars: list[Bar],
    signal_index: int,
    side: int,
    symbol: str,
) -> dict[str, Any] | None:
    entry_index = signal_index + 1 + STRESS_EXECUTION.latency_bars
    exit_index = entry_index + HOLD_HOURS
    if exit_index >= len(bars):
        return None
    entry = bars[entry_index].open
    exit_price = bars[exit_index].close
    if entry <= 0 or not math.isfinite(entry) or not math.isfinite(exit_price):
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


def evaluate_asset(
    bars: list[Bar],
    flow: dict[datetime, dict[str, float]],
    *,
    start: int,
    end: int,
    symbol: str,
    missing_dates: set[date],
) -> list[dict[str, Any]]:
    timestamps = [bar.timestamp for bar in bars]
    last_signal = -10**9
    rows: list[dict[str, Any]] = []
    # Do not score a signal unless its complete trade closes inside this split.
    signal_stop = min(
        len(bars),
        end - 1 - STRESS_EXECUTION.latency_bars - HOLD_HOURS,
    )
    for index in range(start, max(start, signal_stop)):
        side = _signal(flow, timestamps, index, missing_dates)
        if not side or index - last_signal < COOLDOWN_HOURS:
            continue
        result = trade(bars, index, side, symbol)
        if result is not None:
            result["signal_index"] = index
            rows.append(result)
            last_signal = index
    return rows


def _summary_by_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = [[] for _ in range(BLOCKS)]
    for row in rows:
        groups[int(row["block_index"])].append(row)
    pnls = [float(row["net_pnl"]) for row in rows]
    returns = [float(row["net_return"]) for row in rows]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    equity = peak = 0.0
    max_drawdown = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    block_means = [
        statistics.mean(float(row["net_return"]) for row in group) if group else None
        for group in groups
    ]
    valid_means = [value for value in block_means if value is not None]
    median_block = statistics.median(valid_means) if valid_means else None
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_means,
        "net_pnl": sum(pnls),
        "net_return_pct": sum(returns) * 100.0,
        "max_drawdown_pct_of_notional": max_drawdown / NOTIONAL * 100.0,
        "profit_factor": gains / losses if losses else None,
        "execution_cost": sum(float(row["execution_cost"]) for row in rows),
        "median_block_return_to_stress_cost": (
            median_block / (STRESS_EXECUTION.round_trip_bps / 10_000.0)
            if median_block is not None
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
    if bars[0].timestamp > START or bars[-1].timestamp < END - timedelta(hours=1):
        raise ValueError(f"{path} does not fully cover the frozen window")
    selected = [bar for bar in bars if START <= bar.timestamp < END]
    if not selected:
        raise ValueError(f"{path} has no bars in the frozen window")
    return selected


def evaluate_split(
    bars_by_asset: dict[str, list[Bar]],
    flows_by_asset: dict[str, dict[datetime, dict[str, float]]],
    missing_dates_by_asset: dict[str, set[date]],
    start: datetime,
    end: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset, bars in bars_by_asset.items():
        start_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= start), None)
        end_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars))
        if start_index is None:
            raise ValueError(f"{asset} bars do not cover {start.isoformat()}")
        rows.extend(
            evaluate_asset(
                bars,
                flows_by_asset[asset],
                start=start_index,
                end=end_index,
                symbol=asset,
                missing_dates=missing_dates_by_asset[asset],
            )
        )
    rows.sort(key=lambda row: (_utc(row["signal_timestamp"]), row["symbol"]))
    start_epoch = int(start.timestamp() // 3600)
    end_epoch = int(end.timestamp() // 3600)
    width = (end_epoch - start_epoch) / BLOCKS
    for row in rows:
        signal_epoch = int(_utc(row["signal_timestamp"]).timestamp() // 3600)
        row["block_index"] = min(BLOCKS - 1, int((signal_epoch - start_epoch) / width))
    return rows, _summary_by_block(rows)


def _load_liquidation_files(
    liquidations_dir: Path,
) -> tuple[dict[str, dict[datetime, dict[str, float]]], dict[str, set[date]]]:
    flows: dict[str, dict[datetime, dict[str, float]]] = {}
    missing_dates: dict[str, set[date]] = {}
    for asset in ASSETS:
        path = liquidations_dir / LIQUIDATION_FILES[asset]
        flows[asset] = aggregate_hourly(load_liquidations(path))
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_dates[asset] = {
            date.fromisoformat(value) for value in manifest.get("missing_dates", [])
        }
    return flows, missing_dates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--liquidations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bars_by_asset = {
        "BTC": _load_bars(args.btc_path),
        "ETH": _load_bars(args.eth_path),
    }
    flows_by_asset, missing_dates_by_asset = _load_liquidation_files(args.liquidations_dir)
    discovery_rows, discovery = evaluate_split(
        bars_by_asset,
        flows_by_asset,
        missing_dates_by_asset,
        START,
        DISCOVERY_END,
    )
    holdout_rows, holdout = evaluate_split(
        bars_by_asset,
        flows_by_asset,
        missing_dates_by_asset,
        DISCOVERY_END,
        END,
    )
    passes_discovery = _gate(discovery)
    passes_confirmation = bool(passes_discovery and _gate(holdout))
    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": (
            "After an extreme one-hour liquidation flush, forced-flow exhaustion "
            "is followed by an eight-hour reversal."
        ),
        "frozen_parameters": {
            "assets": ["BTCUSDT", "ETHUSDT"],
            "liquidation_source_symbols": ["BTCUSD_PERP", "ETHUSD_PERP"],
            "baseline_hours": BASELINE_HOURS,
            "minimum_baseline_events": MIN_BASELINE_EVENTS,
            "extreme_multiplier": EXTREME_MULTIPLIER,
            "dominance_threshold": DOMINANCE_THRESHOLD,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "direction": "SELL liquidation flow -> long; BUY liquidation flow -> short",
            "position_selection": "BTC and ETH evaluated independently",
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": START.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "end_exclusive": END.isoformat(),
            "holdout_untouched": True,
            "completed_candles_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
            "discovery_trades_cannot_exit_in_holdout": True,
        },
        "source": {
            "provider": "Binance Vision",
            "market": "COIN-M perpetual liquidationSnapshot archives",
            "coverage_constraint": "2023-06-25 through 2024-10-14 inclusive",
            "contract_sizes_usd": {"BTCUSD_PERP": 100.0, "ETHUSD_PERP": 10.0},
            "missing_archive_days_excluded": {
                asset: sorted(value.isoformat() for value in dates)
                for asset, dates in missing_dates_by_asset.items()
            },
        },
        "discovery": discovery,
        "holdout": holdout,
        "passes_discovery": passes_discovery,
        "passes_confirmation": passes_confirmation,
        "status": "confirmed" if passes_confirmation else "not_confirmed",
        "discovery_trade_rows": discovery_rows,
        "holdout_trade_rows": holdout_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "passes_discovery", "passes_confirmation")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
