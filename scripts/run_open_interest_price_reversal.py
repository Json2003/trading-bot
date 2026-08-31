#!/usr/bin/env python3
"""Evaluate the frozen inverse of the open-interest expansion signal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.run_open_interest_price_continuation import (
    ASSETS,
    BLOCKS,
    COOLDOWN_HOURS,
    DISCOVERY_END,
    END,
    HOLD_HOURS,
    MIN_TRADES_PER_BLOCK,
    START,
    _gate,
    _load_bars,
    _signal,
    _summary,
    _trade,
    load_open_interest,
)


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _evaluate(
    bars: list[Any],
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
        original_side, price_move, oi_change = signal
        if index - last_signal_index < COOLDOWN_HOURS:
            continue
        # The only changed rule is direction: continuation is inverted.
        result = _trade(bars, index, -original_side, symbol)
        if result is None:
            continue
        result.update({
            "price_move_1h": price_move,
            "oi_change_6h": oi_change,
            "signal_rule": "price-up plus OI expansion -> short; price-down plus OI expansion -> long",
        })
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

    bars_by_asset = {
        asset: _load_bars(path)
        for asset, path in {"BTC": args.btc_path, "ETH": args.eth_path}.items()
    }
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
            "open interest reverses over the next eight hours."
        ),
        "frozen_parameters": {
            "signal_source": "Binance USD-M futures metrics",
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "price_move_threshold": 0.005,
            "open_interest_measure": "sum_open_interest_value",
            "open_interest_lookback_hours": 6,
            "open_interest_expansion_threshold": 0.01,
            "direction": "price-up plus OI expansion -> short; price-down plus OI expansion -> long",
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "notional": 3000.0,
            "direction_is_inverse_control_of": "open-interest expansion continuation",
        },
        "execution_model": {
            "round_trip_bps": 86,
            "latency_bars": 1,
            "fill_fraction": 0.8,
            "outage_rejection_rate": 0.02,
        },
        "window": {
            "start": START.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "end_exclusive": END.isoformat(),
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
            "market": "USD-M futures metrics plus Binance spot hourly candles",
            "missing_data_is_excluded": True,
            "manifests": manifests,
        },
        "candidates": results,
        "status": "confirmed" if any(
            value["passes_confirmation"] for value in results.values()
        ) else "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        asset: {
            "status": value["status"],
            "discovery_trades": value["discovery"]["trade_count"],
            "holdout_trades": value["holdout"]["trade_count"],
            "discovery_net_return_pct": value["discovery"]["net_return_pct"],
            "holdout_net_return_pct": value["holdout"]["net_return_pct"],
        }
        for asset, value in results.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
