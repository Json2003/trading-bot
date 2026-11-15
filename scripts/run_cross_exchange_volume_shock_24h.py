#!/usr/bin/env python3
"""Evaluate the frozen 24-hour Coinbase volume-shock continuation variant."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
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
        NOTIONAL,
        START,
        _gate,
        _summary,
        _load_binance,
        _utc,
    )
    from scripts.run_cross_exchange_volume_shock import load_coinbase, _signal
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_cross_exchange_lead_lag import (
        ASSETS,
        BLOCKS,
        COOLDOWN_HOURS,
        DISCOVERY_END,
        END,
        NOTIONAL,
        START,
        _gate,
        _summary,
        _load_binance,
        _utc,
    )
    from run_cross_exchange_volume_shock import load_coinbase, _signal

HOLD_HOURS = 24
COINBASE_FILES = {"BTC": "BTCUSD_1h.csv", "ETH": "ETHUSD_1h.csv"}


def _trade(bars: list[Any], signal_index: int, side: int, symbol: str) -> dict[str, Any] | None:
    entry_index = signal_index + 1 + STRESS_EXECUTION.latency_bars
    exit_index = entry_index + HOLD_HOURS
    if exit_index >= len(bars):
        return None
    entry = bars[entry_index].open
    exit_price = bars[exit_index].close
    if entry <= 0 or not math.isfinite(entry) or not math.isfinite(exit_price):
        return None
    gross_return = side * (exit_price / entry - 1.0)
    filled_notional = NOTIONAL * STRESS_EXECUTION.fill_fraction * (
        1.0 - STRESS_EXECUTION.outage_rejection_rate
    )
    trading_cost = filled_notional * STRESS_EXECUTION.round_trip_bps / 10_000.0
    funding_cost = (
        filled_notional * STRESS_EXECUTION.funding_bps_per_bar * HOLD_HOURS / 10_000.0
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
                "coinbase_volume_multiple": current_volume / baseline_volume,
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

    bars_by_asset = {"BTC": _load_binance(args.btc_path), "ETH": _load_binance(args.eth_path)}
    coinbase_by_asset = {
        asset: load_coinbase(args.coinbase_dir / COINBASE_FILES[asset]) for asset in ASSETS
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
        "hypothesis": "A Coinbase directional volume shock continues on Binance for a longer 24-hour horizon.",
        "frozen_parameters": {
            "lead_source": "Coinbase Exchange hourly candles and quote volume",
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "signal_window_hours": 3,
            "volume_lookback_hours": 720,
            "volume_shock_multiplier": 2.0,
            "directional_move_threshold": 0.005,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "direction": "positive Coinbase shock -> long Binance; negative -> short",
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
            "window_is_independent_replication": False,
            "window_note": "Separate pre-registered 24-hour horizon variant; the completed 8-hour result was not overwritten.",
        },
        "source": {
            "lead_provider": "Coinbase Exchange public candles",
            "execution_provider": "Binance Vision spot klines",
            "coinbase_products": ["BTC-USD", "ETH-USD"],
            "binance_symbols": ["BTCUSDT", "ETHUSDT"],
        },
        "candidates": results,
        "status": "confirmed" if any(value["passes_confirmation"] for value in results.values()) else "not_confirmed",
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
