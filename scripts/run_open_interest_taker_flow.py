#!/usr/bin/env python3
"""Evaluate a frozen OI-contraction plus taker-flow continuation hypothesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_open_interest_price_continuation import (
        ASSETS, BLOCKS, COOLDOWN_HOURS, DISCOVERY_END, END, HOLD_HOURS,
        MIN_TRADES_PER_BLOCK, NOTIONAL, START, _gate, _summary, _trade,
        _load_bars,
    )
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_open_interest_price_continuation import (
        ASSETS, BLOCKS, COOLDOWN_HOURS, DISCOVERY_END, END, HOLD_HOURS,
        MIN_TRADES_PER_BLOCK, NOTIONAL, START, _gate, _summary, _trade,
        _load_bars,
    )

OI_CONTRACTION_THRESHOLD = 0.01
PRICE_MOVE_THRESHOLD = 0.005
TAKER_LONG_THRESHOLD = 1.5
TAKER_SHORT_THRESHOLD = 1.0 / TAKER_LONG_THRESHOLD


def load_metrics(path: Path) -> dict[datetime, dict[str, float]]:
    result: dict[datetime, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            oi = float(row["sum_open_interest_value"])
            ratio = float(row["sum_taker_long_short_vol_ratio"])
            if not all(math.isfinite(value) for value in (oi, ratio)) or oi <= 0 or ratio <= 0:
                continue
            result[timestamp] = {"oi": oi, "taker_ratio": ratio}
    return result


def _signal(bars, metrics, index: int) -> tuple[int, float, float, float] | None:
    if index == 0:
        return None
    timestamp = bars[index].timestamp
    prior_timestamp = timestamp - timedelta(hours=6)
    previous_bar = bars[index - 1]
    if (
        previous_bar.timestamp != timestamp - timedelta(hours=1)
        or timestamp not in metrics
        or prior_timestamp not in metrics
    ):
        return None
    current = metrics[timestamp]
    prior = metrics[prior_timestamp]
    price_move = bars[index].close / previous_bar.close - 1.0
    oi_change = current["oi"] / prior["oi"] - 1.0
    ratio = current["taker_ratio"]
    if abs(price_move) < PRICE_MOVE_THRESHOLD or oi_change > -OI_CONTRACTION_THRESHOLD:
        return None
    if price_move > 0 and ratio >= TAKER_LONG_THRESHOLD:
        return 1, price_move, oi_change, ratio
    if price_move < 0 and ratio <= TAKER_SHORT_THRESHOLD:
        return -1, price_move, oi_change, ratio
    return None


def _evaluate(bars, metrics, start, end, symbol):
    start_index = next(i for i, bar in enumerate(bars) if bar.timestamp >= start)
    end_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars))
    rows = []
    last_signal_index = -10**9
    for index in range(start_index, max(start_index, end_index - HOLD_HOURS - 2)):
        signal = _signal(bars, metrics, index)
        if signal is None:
            continue
        side, price_move, oi_change, taker_ratio = signal
        if index - last_signal_index < COOLDOWN_HOURS:
            continue
        result = _trade(bars, index, side, symbol)
        if result is None:
            continue
        result.update({
            "price_move_1h": price_move,
            "oi_change_6h": oi_change,
            "taker_long_short_volume_ratio": taker_ratio,
        })
        rows.append(result)
        last_signal_index = index
    start_epoch = int(start.timestamp() // 3600)
    width = (end - start).total_seconds() / 3600 / BLOCKS
    for row in rows:
        signal_epoch = _utc(row["signal_timestamp"]).timestamp() // 3600
        row["block_index"] = min(BLOCKS - 1, int((signal_epoch - start_epoch) / width))
    return rows, _summary(rows)


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bars = {
        asset: _load_bars(path)
        for asset, path in {"BTC": args.btc_path, "ETH": args.eth_path}.items()
    }
    metrics = {
        asset: load_metrics(args.metrics_dir / f"{symbol}_1h.csv")
        for asset, symbol in ASSETS.items()
    }
    candidates: dict[str, Any] = {}
    for asset in ASSETS:
        discovery_rows, discovery = _evaluate(bars[asset], metrics[asset], START, DISCOVERY_END, asset)
        holdout_rows, holdout = _evaluate(bars[asset], metrics[asset], DISCOVERY_END, END, asset)
        candidates[asset] = {
            "discovery": discovery, "holdout": holdout,
            "passes_discovery": _gate(discovery),
            "passes_confirmation": bool(_gate(discovery) and _gate(holdout)),
            "status": "confirmed" if _gate(discovery) and _gate(holdout) else "not_confirmed",
            "discovery_trade_rows": discovery_rows, "holdout_trade_rows": holdout_rows,
        }
    manifests = {}
    for asset, symbol in ASSETS.items():
        manifests[asset] = json.loads(
            (args.metrics_dir / f"{symbol}_1h.manifest.json").read_text(encoding="utf-8")
        )
    report = {
        "schema_version": 1, "research_only": True,
        "paper_orders_placed": False, "live_orders_placed": False,
        "leverage_enabled": False, "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": (
            "A completed price move accompanied by six-hour open-interest "
            "contraction and aligned aggressive taker flow continues for eight hours."
        ),
        "frozen_parameters": {
            "signal_source": "Binance USD-M futures metrics",
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "price_move_threshold": PRICE_MOVE_THRESHOLD,
            "oi_measure": "sum_open_interest_value",
            "oi_lookback_hours": 6,
            "oi_contraction_threshold": OI_CONTRACTION_THRESHOLD,
            "taker_flow_measure": "sum_taker_long_short_vol_ratio",
            "taker_long_threshold": TAKER_LONG_THRESHOLD,
            "taker_short_threshold": TAKER_SHORT_THRESHOLD,
            "direction": "aligned price and taker flow -> same-direction trade",
            "hold_hours": HOLD_HOURS, "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "notional": NOTIONAL,
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": START.isoformat(), "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "end_exclusive": END.isoformat(), "evaluation_length_days": 366,
            "holdout_untouched": True, "completed_candles_only": True,
            "six_chronological_blocks_per_split": True, "holdout_selection_used": False,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "newest_unseen_data_used": False, "window_is_independent_replication": False,
        },
        "source": {
            "provider": "Binance Vision", "market": "USD-M futures metrics",
            "archive_period": "5-minute metrics downsampled to latest completed observation per hour",
            "manifests": manifests, "missing_data_is_excluded": True,
        },
        "candidates": candidates,
        "status": "confirmed" if any(v["passes_confirmation"] for v in candidates.values()) else "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        asset: {"status": value["status"], "discovery_trades": value["discovery"]["trade_count"],
                "holdout_trades": value["holdout"]["trade_count"]}
        for asset, value in candidates.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
