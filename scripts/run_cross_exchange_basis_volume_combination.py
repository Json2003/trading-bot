#!/usr/bin/env python3
"""Evaluate a frozen cross-exchange basis-dislocation plus volume confirmation hypothesis."""

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
        _utc,
    )
    from scripts.run_momentum_volatility_research import Bar, load_bars
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
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
        _utc,
    )
    from run_momentum_volatility_research import Bar, load_bars

# One-year frozen evaluation window: six months discovery, six months holdout.
START = datetime(2023, 10, 15, tzinfo=timezone.utc)
DISCOVERY_END = datetime(2024, 4, 15, tzinfo=timezone.utc)
END = datetime(2024, 10, 15, tzinfo=timezone.utc)
BASIS_LOOKBACK_HOURS = 720
BASIS_DEVIATION_THRESHOLD = 0.002
CONFIRMATION_HOURS = 3
VOLUME_LOOKBACK_HOURS = 720
VOLUME_SHOCK_MULTIPLIER = 2.0
COINBASE_FILES = {"BTC": "BTCUSD_1h.csv", "ETH": "ETHUSD_1h.csv"}
WARMUP_START = START - timedelta(hours=BASIS_LOOKBACK_HOURS + CONFIRMATION_HOURS + 2)


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


def _hours_ending(timestamp: datetime, count: int) -> list[datetime]:
    return [
        timestamp - timedelta(hours=offset)
        for offset in range(count - 1, -1, -1)
    ]


def _basis_signal(
    coinbase: dict[datetime, dict[str, float]],
    binance_close: dict[datetime, float],
    timestamp: datetime,
) -> tuple[int, float, float, float, float] | None:
    current_hours = _hours_ending(timestamp, CONFIRMATION_HOURS)
    prior_hours = _hours_ending(
        timestamp - timedelta(hours=CONFIRMATION_HOURS),
        BASIS_LOOKBACK_HOURS,
    )
    required = current_hours + prior_hours
    if any(hour not in coinbase or hour not in binance_close for hour in required):
        return None

    basis = {
        hour: coinbase[hour]["close"] / binance_close[hour] - 1.0
        for hour in required
    }
    baseline_values = [basis[hour] for hour in prior_hours]
    baseline = statistics.median(baseline_values)
    deviations = [basis[hour] - baseline for hour in current_hours]
    current_deviation = deviations[-1]
    if not all(abs(value) >= BASIS_DEVIATION_THRESHOLD for value in deviations):
        return 0, current_deviation, baseline, 0.0, 0.0
    if not all(value > 0 for value in deviations) and not all(value < 0 for value in deviations):
        return 0, current_deviation, baseline, 0.0, 0.0

    current_volume = sum(coinbase[hour]["volume"] for hour in current_hours)
    prior_volume_windows = []
    for offset in range(0, VOLUME_LOOKBACK_HOURS - CONFIRMATION_HOURS + 1):
        window = _hours_ending(
            timestamp - timedelta(hours=CONFIRMATION_HOURS + offset),
            CONFIRMATION_HOURS,
        )
        prior_volume_windows.append(sum(coinbase[hour]["volume"] for hour in window))
    volume_baseline = statistics.median(prior_volume_windows)
    volume_multiple = (
        current_volume / volume_baseline if volume_baseline > 0 else 0.0
    )
    if volume_baseline <= 0 or current_volume < VOLUME_SHOCK_MULTIPLIER * volume_baseline:
        return 0, current_deviation, baseline, volume_multiple, current_volume

    side = 1 if current_deviation > 0 else -1
    return side, current_deviation, baseline, volume_multiple, current_volume


def _load_binance(path: Path) -> list[Bar]:
    bars = load_bars(path)
    if not bars or bars[0].timestamp > WARMUP_START or bars[-1].timestamp < END:
        raise ValueError(f"{path} does not cover the frozen one-year window and warmup")
    return [bar for bar in bars if WARMUP_START <= bar.timestamp < END]


def _evaluate_asset(
    bars: list[Bar],
    coinbase: dict[datetime, dict[str, float]],
    start: datetime,
    end: datetime,
    symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    binance_close = {bar.timestamp: bar.close for bar in bars}
    start_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= start), None)
    end_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars))
    if start_index is None:
        raise ValueError(f"{symbol} bars do not cover {start.isoformat()}")

    rows: list[dict[str, Any]] = []
    last_signal_index = -10**9
    for index in range(start_index, max(start_index, end_index - HOLD_HOURS - 2)):
        timestamp = bars[index].timestamp
        signal = _basis_signal(coinbase, binance_close, timestamp)
        if signal is None:
            continue
        side, deviation, baseline, volume_multiple, current_volume = signal
        if not side or index - last_signal_index < COOLDOWN_HOURS:
            continue
        result = _trade(bars, index, side, symbol)
        if result is None:
            continue
        result.update(
            {
                "basis_deviation": deviation,
                "basis_baseline": baseline,
                "coinbase_volume_3h": current_volume,
                "coinbase_volume_multiple": volume_multiple,
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
            "status": (
                "confirmed"
                if _gate(discovery) and _gate(holdout)
                else "not_confirmed"
            ),
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
            "A persistent Coinbase-vs-Binance basis dislocation, confirmed by "
            "a Coinbase volume shock, converges on Binance over the next eight hours."
        ),
        "frozen_parameters": {
            "lead_source": "Coinbase Exchange hourly close and base volume",
            "execution_source": "Binance spot hourly close/open",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "basis_definition": "Coinbase close / Binance close - 1",
            "basis_baseline": "median of prior 720 completed hourly basis observations",
            "basis_deviation_threshold": BASIS_DEVIATION_THRESHOLD,
            "confirmation_hours": CONFIRMATION_HOURS,
            "volume_lookback_hours": VOLUME_LOOKBACK_HOURS,
            "volume_shock_multiplier": VOLUME_SHOCK_MULTIPLIER,
            "direction": (
                "Coinbase premium -> long Binance; Coinbase discount -> short Binance"
            ),
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
            "window_note": (
                "New pre-specified combination tested on a one-year historical "
                "window; no thresholds were changed after evaluation."
            ),
        },
        "source": {
            "lead_provider": "Coinbase Exchange public candles",
            "execution_provider": "Binance Vision spot klines",
            "coinbase_products": ["BTC-USD", "ETH-USD"],
            "binance_symbols": ["BTCUSDT", "ETHUSDT"],
        },
        "candidates": results,
        "status": (
            "confirmed"
            if any(value["passes_confirmation"] for value in results.values())
            else "not_confirmed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
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
