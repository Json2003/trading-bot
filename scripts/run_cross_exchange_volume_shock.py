#!/usr/bin/env python3
"""Evaluate one frozen Coinbase-volume-shock to Binance continuation hypothesis."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

try:
    from scripts.run_cross_exchange_lead_lag import (
        ASSETS,
        BLOCKS,
        COOLDOWN_HOURS,
        DISCOVERY_END,
        END,
        HOLD_HOURS,
        MIN_TRADES_PER_BLOCK,
        NOTIONAL,
        START,
        _gate,
        _summary,
        _trade,
        _load_binance,
        _utc,
    )
except ModuleNotFoundError:
    from run_cross_exchange_lead_lag import (
        ASSETS,
        BLOCKS,
        COOLDOWN_HOURS,
        DISCOVERY_END,
        END,
        HOLD_HOURS,
        MIN_TRADES_PER_BLOCK,
        NOTIONAL,
        START,
        _gate,
        _summary,
        _trade,
        _load_binance,
        _utc,
    )

VOLUME_LOOKBACK_HOURS = 720
VOLUME_SHOCK_MULTIPLIER = 2.0
MOVE_THRESHOLD = 0.005
SIGNAL_WINDOW_HOURS = 3
COINBASE_FILES = {"BTC": "BTCUSD_1h.csv", "ETH": "ETHUSD_1h.csv"}


def load_coinbase(path: Path) -> dict[datetime, dict[str, float]]:
    result: dict[datetime, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _utc(row["timestamp"])
            close = float(row["close"])
            volume = float(row["volume"])
            if (
                not math.isfinite(close)
                or close <= 0
                or not math.isfinite(volume)
                or volume < 0
            ):
                raise ValueError("Coinbase close and volume must be finite and valid")
            result[timestamp] = {"close": close, "volume": volume}
    return result


def _hour_range(end: datetime, count: int) -> list[datetime]:
    return [
        end - timedelta(hours=offset)
        for offset in range(count - 1, -1, -1)
    ]


def _signal(
    coinbase: dict[datetime, dict[str, float]],
    timestamp: datetime,
) -> tuple[int, float, float, float] | None:
    current_hours = _hour_range(timestamp, SIGNAL_WINDOW_HOURS)
    prior_hours = _hour_range(timestamp - timedelta(hours=SIGNAL_WINDOW_HOURS), VOLUME_LOOKBACK_HOURS)
    required = current_hours + prior_hours
    if any(hour not in coinbase for hour in required):
        return None
    start = coinbase[current_hours[0]]["close"]
    end = coinbase[current_hours[-1]]["close"]
    move = end / start - 1.0
    current_volume = sum(coinbase[hour]["volume"] for hour in current_hours)
    prior_windows = []
    for offset in range(0, VOLUME_LOOKBACK_HOURS - SIGNAL_WINDOW_HOURS + 1):
        window = _hour_range(
            timestamp - timedelta(hours=SIGNAL_WINDOW_HOURS + offset),
            SIGNAL_WINDOW_HOURS,
        )
        prior_windows.append(sum(coinbase[hour]["volume"] for hour in window))
    baseline = statistics.median(prior_windows)
    if baseline <= 0 or current_volume < VOLUME_SHOCK_MULTIPLIER * baseline:
        return 0, move, current_volume, baseline
    if move >= MOVE_THRESHOLD:
        return 1, move, current_volume, baseline
    if move <= -MOVE_THRESHOLD:
        return -1, move, current_volume, baseline
    return 0, move, current_volume, baseline


def _evaluate_asset(
    bars: list[Any],
    coinbase: dict[datetime, dict[str, float]],
    start: datetime,
    end: datetime,
    symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= start), None)
    end_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars))
    if start_index is None:
        raise ValueError(f"{symbol} bars do not cover {start.isoformat()}")
    rows: list[dict[str, Any]] = []
    last_signal_index = -10**9
    signal_stop = max(start_index, end_index - HOLD_HOURS - 2)
    for index in range(start_index, signal_stop):
        signal = _signal(coinbase, bars[index].timestamp)
        if signal is None:
            continue
        side, move, current_volume, baseline_volume = signal
        if not side or index - last_signal_index < COOLDOWN_HOURS:
            continue
        result = _trade(bars, index, side, symbol)
        if result is None:
            continue
        result.update(
            {
                "coinbase_move": move,
                "coinbase_volume_3h": current_volume,
                "coinbase_volume_baseline_3h": baseline_volume,
                "coinbase_volume_multiple": (
                    current_volume / baseline_volume if baseline_volume else None
                ),
            }
        )
        rows.append(result)
        last_signal_index = index
    rows.sort(key=lambda row: (_utc(row["signal_timestamp"]), row["symbol"]))
    start_epoch = int(start.timestamp() // 3600)
    end_epoch = int(end.timestamp() // 3600)
    width = (end_epoch - start_epoch) / BLOCKS
    for row in rows:
        signal_epoch = int(_utc(row["signal_timestamp"]).timestamp() // 3600)
        row["block_index"] = min(BLOCKS - 1, int((signal_epoch - start_epoch) / width))
    return rows, _summary(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--coinbase-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bars_by_asset = {
        "BTC": _load_binance(args.btc_path),
        "ETH": _load_binance(args.eth_path),
    }
    coinbase_by_asset = {
        asset: load_coinbase(args.coinbase_dir / COINBASE_FILES[asset])
        for asset in ASSETS
    }
    results: dict[str, Any] = {}
    for asset, bars in bars_by_asset.items():
        discovery_rows, discovery = _evaluate_asset(
            bars, coinbase_by_asset[asset], START, DISCOVERY_END, asset
        )
        holdout_rows, holdout = _evaluate_asset(
            bars, coinbase_by_asset[asset], DISCOVERY_END, END, asset
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

    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": (
            "A large completed Coinbase volume shock with directional movement "
            "leads to continuation on Binance during the next eight hours."
        ),
        "frozen_parameters": {
            "lead_source": "Coinbase Exchange hourly candles and quote volume",
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "signal_window_hours": SIGNAL_WINDOW_HOURS,
            "volume_lookback_hours": VOLUME_LOOKBACK_HOURS,
            "volume_shock_multiplier": VOLUME_SHOCK_MULTIPLIER,
            "directional_move_threshold": MOVE_THRESHOLD,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "direction": "positive Coinbase shock -> long Binance; negative -> short",
            "notional": NOTIONAL,
        },
        "execution_model": __import__(
            "scripts.execution_model", fromlist=["STRESS_EXECUTION"]
        ).STRESS_EXECUTION.as_dict(),
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
            "window_is_independent_replication": False,
            "window_note": "New pre-specified volume-shock rule; no thresholds were changed after evaluation.",
        },
        "source": {
            "lead_provider": "Coinbase Exchange public candles",
            "execution_provider": "Binance Vision spot klines",
            "coinbase_products": ["BTC-USD", "ETH-USD"],
            "binance_symbols": ["BTCUSDT", "ETHUSDT"],
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
                asset: {
                    "status": value["status"],
                    "discovery_trades": value["discovery"]["trade_count"],
                    "holdout_trades": value["holdout"]["trade_count"],
                }
                for asset, value in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
