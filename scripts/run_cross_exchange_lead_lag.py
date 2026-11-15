#!/usr/bin/env python3
"""Evaluate one frozen Coinbase-leads-Binance hourly lead/lag hypothesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
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
ASSETS = {"BTC": ("BTCUSDT", "BTCUSD"), "ETH": ("ETHUSDT", "ETHUSD")}
LEAD_HOURS = 3
LEAD_MOVE_THRESHOLD = 0.010
MINIMUM_LEAD_LAG_GAP = 0.005
HOLD_HOURS = 8
COOLDOWN_HOURS = 8
NOTIONAL = 3_000.0
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20
COINBASE_FILES = {"BTC": "BTCUSD_1h.csv", "ETH": "ETHUSD_1h.csv"}


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_coinbase(path: Path) -> dict[datetime, float]:
    result: dict[datetime, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _utc(row["timestamp"])
            close = float(row["close"])
            if not math.isfinite(close) or close <= 0:
                raise ValueError("Coinbase close must be finite and positive")
            result[timestamp] = close
    return result


def _signal(
    coinbase: dict[datetime, float],
    binance: dict[datetime, float],
    timestamp: datetime,
) -> tuple[int, float, float, float] | None:
    prior_timestamp = timestamp.fromtimestamp(
        timestamp.timestamp() - LEAD_HOURS * 3600,
        tz=timezone.utc,
    )
    lead_start = coinbase.get(prior_timestamp)
    lead_end = coinbase.get(timestamp)
    execution_start = binance.get(prior_timestamp)
    execution_end = binance.get(timestamp)
    if None in {lead_start, lead_end, execution_start, execution_end}:
        return None
    lead_move = lead_end / lead_start - 1.0
    execution_move = execution_end / execution_start - 1.0
    gap = lead_move - execution_move
    if lead_move >= LEAD_MOVE_THRESHOLD and gap >= MINIMUM_LEAD_LAG_GAP:
        return 1, lead_move, execution_move, gap
    if lead_move <= -LEAD_MOVE_THRESHOLD and gap <= -MINIMUM_LEAD_LAG_GAP:
        return -1, lead_move, execution_move, gap
    return 0, lead_move, execution_move, gap


def _trade(bars: list[Bar], signal_index: int, side: int, symbol: str) -> dict[str, Any] | None:
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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _load_binance(path: Path) -> list[Bar]:
    bars = load_bars(path)
    if not bars or bars[0].timestamp > START or bars[-1].timestamp < END:
        raise ValueError(f"{path} does not cover the frozen window")
    return [bar for bar in bars if START <= bar.timestamp < END]


def _evaluate_split(
    bars: list[Bar],
    coinbase: dict[datetime, float],
    start: datetime,
    end: datetime,
    symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bar_by_timestamp = {bar.timestamp: bar.close for bar in bars}
    start_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= start), None)
    end_index = next((i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars))
    if start_index is None:
        raise ValueError(f"{symbol} bars do not cover {start.isoformat()}")
    rows: list[dict[str, Any]] = []
    last_signal_index = -10**9
    for index in range(start_index, max(start_index, end_index - HOLD_HOURS - 2)):
        timestamp = bars[index].timestamp
        signal = _signal(coinbase, bar_by_timestamp, timestamp)
        if signal is None:
            continue
        side, lead_move, execution_move, gap = signal
        if not side or index - last_signal_index < COOLDOWN_HOURS:
            continue
        result = _trade(bars, index, side, symbol)
        if result is not None:
            result.update(
                {
                    "lead_move": lead_move,
                    "execution_move": execution_move,
                    "lead_lag_gap": gap,
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
        discovery_rows, discovery = _evaluate_split(
            bars, coinbase_by_asset[asset], START, DISCOVERY_END, asset
        )
        holdout_rows, holdout = _evaluate_split(
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
            "A large completed Coinbase move leads an under-reacting Binance "
            "market, which catches up in the next eight hours."
        ),
        "frozen_parameters": {
            "lead_source": "Coinbase Exchange hourly candles",
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "lead_window_hours": LEAD_HOURS,
            "lead_move_threshold": LEAD_MOVE_THRESHOLD,
            "minimum_lead_lag_gap": MINIMUM_LEAD_LAG_GAP,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "direction": "positive Coinbase lead and positive gap -> long Binance; negative -> short",
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
            "window_note": "New cross-exchange data source tested on the same frozen historical window; this is not an independent time-period replication of earlier hypotheses.",
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
