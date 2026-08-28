#!/usr/bin/env python3
"""Evaluate one frozen book-imbalance plus liquidation confirmation experiment."""

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
    from scripts.run_liquidation_flow_reversal import (
        ASSETS,
        BASELINE_HOURS,
        COOLDOWN_HOURS,
        DOMINANCE_THRESHOLD,
        END as LIQ_END,
        EXTREME_MULTIPLIER,
        HOLD_HOURS,
        LIQUIDATION_FILES,
        MIN_BASELINE_EVENTS,
        NOTIONAL,
        START as LIQ_START,
        DISCOVERY_END as LIQ_DISCOVERY_END,
        _signal as liquidation_signal,
        aggregate_hourly,
        load_liquidations,
        trade,
    )
    from scripts.run_momentum_volatility_research import Bar, load_bars
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_liquidation_flow_reversal import (
        ASSETS,
        BASELINE_HOURS,
        COOLDOWN_HOURS,
        DOMINANCE_THRESHOLD,
        END as LIQ_END,
        EXTREME_MULTIPLIER,
        HOLD_HOURS,
        LIQUIDATION_FILES,
        MIN_BASELINE_EVENTS,
        NOTIONAL,
        START as LIQ_START,
        DISCOVERY_END as LIQ_DISCOVERY_END,
        _signal as liquidation_signal,
        aggregate_hourly,
        load_liquidations,
        trade,
    )
    from run_momentum_volatility_research import Bar, load_bars

START = datetime(2025, 8, 1, tzinfo=timezone.utc)
DISCOVERY_END = datetime(2026, 2, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 1, tzinfo=timezone.utc)
BOOK_IMBALANCE_THRESHOLD = 0.20
BOOK_PERSISTENCE_HOURS = 3
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20
BOOK_FILES = {
    "BTC": "BTCUSDT_bookdepth_1h.csv",
    "ETH": "ETHUSDT_bookdepth_1h.csv",
}


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_bookdepth(path: Path) -> dict[datetime, float]:
    result: dict[datetime, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _utc(row["timestamp"])
            imbalance = float(row["imbalance"])
            snapshots = int(row["snapshot_count"])
            if not math.isfinite(imbalance) or not -1 <= imbalance <= 1:
                raise ValueError("book imbalance must be finite and within [-1, 1]")
            if snapshots <= 0:
                raise ValueError("book snapshot_count must be positive")
            result[timestamp] = imbalance
    return result


def book_side(book: dict[datetime, float], timestamps: list[datetime], index: int) -> int:
    start = max(0, index - BOOK_PERSISTENCE_HOURS + 1)
    values = [book.get(timestamp) for timestamp in timestamps[start : index + 1]]
    if len(values) != BOOK_PERSISTENCE_HOURS or any(value is None for value in values):
        return 0
    numeric = [float(value) for value in values if value is not None]
    if all(value >= BOOK_IMBALANCE_THRESHOLD for value in numeric):
        return 1
    if all(value <= -BOOK_IMBALANCE_THRESHOLD for value in numeric):
        return -1
    return 0


def candidate_side(
    candidate: str,
    flow: dict[datetime, dict[str, float]],
    book: dict[datetime, float],
    timestamps: list[datetime],
    index: int,
    missing_dates: set[date],
) -> int:
    pressure = book_side(book, timestamps, index)
    if candidate == "orderbook_only":
        return pressure
    if candidate != "combined":
        raise ValueError(f"unknown candidate: {candidate}")
    forced_flow = liquidation_signal(flow, timestamps, index, missing_dates)
    return forced_flow if forced_flow and forced_flow == pressure else 0


def evaluate_asset(
    bars: list[Bar],
    flow: dict[datetime, dict[str, float]],
    book: dict[datetime, float],
    *,
    start: int,
    end: int,
    symbol: str,
    missing_dates: set[date],
    candidate: str,
) -> list[dict[str, Any]]:
    timestamps = [bar.timestamp for bar in bars]
    last_signal = -10**9
    rows: list[dict[str, Any]] = []
    signal_stop = min(len(bars), end - 1 - STRESS_EXECUTION.latency_bars - HOLD_HOURS)
    for index in range(start, max(start, signal_stop)):
        side = candidate_side(candidate, flow, book, timestamps, index, missing_dates)
        if not side or index - last_signal < COOLDOWN_HOURS:
            continue
        result = trade(bars, index, side, symbol)
        if result is not None:
            result["signal_index"] = index
            result["candidate"] = candidate
            result["book_imbalance"] = book.get(timestamps[index])
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
    mean_return = statistics.mean(returns) if returns else None
    volatility = statistics.pstdev(returns) if len(returns) > 1 else None
    sharpe_proxy = (
        mean_return / volatility * math.sqrt(len(returns))
        if mean_return is not None and volatility and volatility > 0
        else None
    )
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_means,
        "net_pnl": sum(pnls),
        "net_return_pct": sum(returns) * 100.0,
        "max_drawdown_pct_of_notional": max_drawdown / NOTIONAL * 100.0,
        "sharpe_proxy": sharpe_proxy,
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
    books_by_asset: dict[str, dict[datetime, float]],
    missing_dates_by_asset: dict[str, set[date]],
    start: datetime,
    end: datetime,
    candidate: str,
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
                books_by_asset[asset],
                start=start_index,
                end=end_index,
                symbol=asset,
                missing_dates=missing_dates_by_asset[asset],
                candidate=candidate,
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


def _load_liquidation_files(liquidations_dir: Path) -> tuple[
    dict[str, dict[datetime, dict[str, float]]], dict[str, set[date]]
]:
    flows: dict[str, dict[datetime, dict[str, float]]] = {}
    missing_dates: dict[str, set[date]] = {}
    for asset in ASSETS:
        path = liquidations_dir / LIQUIDATION_FILES[asset]
        flows[asset] = aggregate_hourly(load_liquidations(path))
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        missing_dates[asset] = {
            date.fromisoformat(value) for value in manifest.get("missing_dates", [])
        }
    return flows, missing_dates


def _load_book_files(bookdepth_dir: Path) -> tuple[dict[str, dict[datetime, float]], dict[str, list[str]]]:
    books: dict[str, dict[datetime, float]] = {}
    missing_dates: dict[str, list[str]] = {}
    for asset in ASSETS:
        path = bookdepth_dir / BOOK_FILES[asset]
        books[asset] = load_bookdepth(path)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        missing_dates[asset] = list(manifest.get("missing_dates", []))
    return books, missing_dates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--liquidations-dir", type=Path, required=True)
    parser.add_argument("--bookdepth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bars_by_asset = {"BTC": _load_bars(args.btc_path), "ETH": _load_bars(args.eth_path)}
    flows_by_asset, liquidation_missing = _load_liquidation_files(args.liquidations_dir)
    books_by_asset, book_missing = _load_book_files(args.bookdepth_dir)
    results: dict[str, Any] = {}
    for candidate in ("orderbook_only", "combined"):
        discovery_rows, discovery = evaluate_split(
            bars_by_asset,
            flows_by_asset,
            books_by_asset,
            liquidation_missing,
            START,
            DISCOVERY_END,
            candidate,
        )
        holdout_rows, holdout = evaluate_split(
            bars_by_asset,
            flows_by_asset,
            books_by_asset,
            liquidation_missing,
            DISCOVERY_END,
            END,
            candidate,
        )
        results[candidate] = {
            "discovery": discovery,
            "holdout": holdout,
            "passes_discovery": _gate(discovery),
            "passes_confirmation": bool(_gate(discovery) and _gate(holdout)),
            "status": "confirmed" if _gate(discovery) and _gate(holdout) else "not_confirmed",
            "discovery_trade_rows": discovery_rows,
            "holdout_trade_rows": holdout_rows,
        }

    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": (
            "A persistent one-sided order-book imbalance identifies near-term "
            "direction, and liquidation-flow reversal is valid only when the "
            "order book confirms the same direction."
        ),
        "frozen_parameters": {
            "assets": ["BTCUSDT", "ETHUSDT"],
            "book_depth_source": "Binance Vision USD-M daily bookDepth",
            "book_band_percentage": 1,
            "book_imbalance": "(bid_notional_at_-1pct - ask_notional_at_+1pct) / sum",
            "book_imbalance_threshold": BOOK_IMBALANCE_THRESHOLD,
            "book_persistence_hours": BOOK_PERSISTENCE_HOURS,
            "combined_rule": "liquidation reversal side must equal persistent book-pressure side",
            "liquidation_baseline_hours": BASELINE_HOURS,
            "liquidation_minimum_baseline_events": MIN_BASELINE_EVENTS,
            "liquidation_extreme_multiplier": EXTREME_MULTIPLIER,
            "liquidation_dominance_threshold": DOMINANCE_THRESHOLD,
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
            "holdout_untouched": True,
            "completed_candles_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "newest_unseen_data_used": False,
        },
        "source": {
            "provider": "Binance Vision",
            "bookdepth_archives": "USD-M daily bookDepth",
            "liquidation_archives": "COIN-M daily liquidationSnapshot",
            "bookdepth_missing_archive_days": book_missing,
            "liquidation_missing_archive_days": {
                asset: sorted(value) for asset, value in liquidation_missing.items()
            },
        },
        "candidates": results,
        "status": "confirmed"
        if any(value["passes_confirmation"] for value in results.values())
        else "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                candidate: {
                    "status": value["status"],
                    "discovery_trades": value["discovery"]["trade_count"],
                    "holdout_trades": value["holdout"]["trade_count"],
                }
                for candidate, value in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
