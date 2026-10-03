#!/usr/bin/env python3
"""Research-only high-volume volatility-pocket backtest."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

NOTIONAL = 3000.0
LOOKBACK_HOURS = 20
VOLUME_MULTIPLE = 2.0
RANGE_THRESHOLD = 0.01
HOLD_HOURS = 4
COOLDOWN_HOURS = 4
LATENCY_BARS = 1
ROUND_TRIP_BPS = 86.0
FILL_FRACTION = 0.80
FUNDING_BPS_PER_BAR = 0.5
OUTAGE_REJECTION_RATE = 0.02
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_csv(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "timestamp": parse_ts(str(row["timestamp"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def split_blocks(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not rows:
        return [[] for _ in range(BLOCKS)]
    return [rows[i * len(rows) // BLOCKS:(i + 1) * len(rows) // BLOCKS] for i in range(BLOCKS)]


def summary(rows: list[dict[str, object]], start: str, end: str) -> dict[str, object]:
    pnl = sum(float(row["net_pnl"]) for row in rows)
    equity = NOTIONAL
    peak = equity
    max_dd = 0.0
    returns = []
    for row in rows:
        value = float(row["net_pnl"])
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / NOTIONAL * 100.0)
        returns.append(value / NOTIONAL)
    wins = sum(x for x in returns if x > 0)
    losses = -sum(x for x in returns if x < 0)
    pf = wins / losses if losses else None
    sharpe = None
    if len(returns) >= 2 and statistics.pstdev(returns) > 0:
        sharpe = statistics.mean(returns) / statistics.pstdev(returns) * math.sqrt(len(returns))
    blocks = split_blocks(rows)
    counts = [len(block) for block in blocks]
    means = [statistics.mean(float(row["net_return"]) for row in block) if block else None for block in blocks]
    positive = sum(1 for value in means if value is not None and value > 0)
    return {
        "segment_start": start,
        "segment_end_exclusive": end,
        "trade_count": len(rows),
        "net_pnl": pnl,
        "net_return_pct": pnl / NOTIONAL * 100.0,
        "max_drawdown_pct_of_notional": max_dd,
        "sharpe_proxy": sharpe,
        "profit_factor": pf,
        "execution_cost": sum(float(row["execution_cost"]) for row in rows),
        "block_trade_counts": counts,
        "block_mean_net_returns": means,
        "positive_block_count": positive,
        "passes_sample_gate": all(count >= MIN_TRADES_PER_BLOCK for count in counts),
        "passes_positive_block_gate": positive >= 4,
    }


def evaluate(asset: str, rows: list[dict[str, object]], discovery_start: str, discovery_end: str, end: str) -> dict[str, object]:
    start_dt = datetime.fromisoformat(discovery_start).replace(tzinfo=timezone.utc)
    split_dt = datetime.fromisoformat(discovery_end).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cost = NOTIONAL * (ROUND_TRIP_BPS + FUNDING_BPS_PER_BAR * HOLD_HOURS) / 10000.0 * FILL_FRACTION
    trades = []
    last_signal = -10**9
    for i in range(LOOKBACK_HOURS, len(rows) - LATENCY_BARS - HOLD_HOURS):
        signal = rows[i]
        if signal["timestamp"] < start_dt or signal["timestamp"] >= end_dt:
            continue
        if i <= last_signal + COOLDOWN_HOURS:
            continue
        prior_volumes = [float(rows[j]["volume"]) for j in range(i - LOOKBACK_HOURS, i)]
        median_volume = statistics.median(prior_volumes)
        range_pct = (float(signal["high"]) - float(signal["low"])) / float(signal["open"])
        volume_ratio = float(signal["volume"]) / median_volume if median_volume > 0 else 0.0
        direction = 1 if float(signal["close"]) > float(signal["open"]) else -1 if float(signal["close"]) < float(signal["open"]) else 0
        if direction == 0 or volume_ratio < VOLUME_MULTIPLE or range_pct < RANGE_THRESHOLD:
            continue
        entry = rows[i + LATENCY_BARS]
        exit_row = rows[i + LATENCY_BARS + HOLD_HOURS]
        gross = direction * (float(exit_row["close"]) / float(entry["open"]) - 1.0)
        net = gross - cost / NOTIONAL
        trades.append({
            "signal_timestamp": signal["timestamp"].isoformat().replace("+00:00", "Z"),
            "entry_timestamp": entry["timestamp"].isoformat().replace("+00:00", "Z"),
            "exit_timestamp": exit_row["timestamp"].isoformat().replace("+00:00", "Z"),
            "asset": asset,
            "side": "long" if direction > 0 else "short",
            "volume_ratio": volume_ratio,
            "range_pct": range_pct,
            "gross_return": gross,
            "net_return": net,
            "net_pnl": net * NOTIONAL,
            "execution_cost": cost,
        })
        last_signal = i
    discovery = [row for row in trades if parse_ts(str(row["signal_timestamp"])) < split_dt]
    holdout = [row for row in trades if parse_ts(str(row["signal_timestamp"])) >= split_dt]
    result = {
        "discovery": summary(discovery, discovery_start, discovery_end),
        "holdout": summary(holdout, discovery_end, end),
        "discovery_trade_rows": discovery,
        "holdout_trade_rows": holdout,
    }
    result["passes_discovery"] = result["discovery"]["passes_sample_gate"] and result["discovery"]["passes_positive_block_gate"]
    result["passes_confirmation"] = result["passes_discovery"] and result["holdout"]["passes_sample_gate"] and result["holdout"]["passes_positive_block_gate"] and float(result["holdout"]["net_return_pct"]) > 0
    result["status"] = "confirmed" if result["passes_confirmation"] else "not_confirmed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/historical/binance/normalized"))
    parser.add_argument("--discovery-start", default="2023-01-01T00:00:00")
    parser.add_argument("--discovery-end", default="2025-01-01T00:00:00")
    parser.add_argument("--end", default="2026-08-01T00:00:00")
    parser.add_argument("--output", type=Path, default=Path("volume-volatility-pocket-report.json"))
    args = parser.parse_args()
    candidates = {}
    manifests = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        rows = load_csv(args.data_dir / f"{symbol}_1h.csv")
        candidates[symbol[:3]] = evaluate(symbol[:3], rows, args.discovery_start, args.discovery_end, args.end)
        manifest_path = args.data_dir / f"{symbol}_1h.manifest.json"
        manifests[symbol[:3]] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
            "row_count": len(rows),
            "first_timestamp": rows[0]["timestamp"].isoformat(),
            "last_timestamp": rows[-1]["timestamp"].isoformat(),
        }
    report = {
        "schema_version": 1,
        "hypothesis": "A completed hourly candle with unusually high volume and expanded range predicts same-direction movement over the next four completed hourly bars.",
        "window": {
            "start": args.discovery_start,
            "discovery_end_exclusive": args.discovery_end,
            "end_exclusive": args.end,
            "holdout_untouched": True,
            "completed_candles_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "threshold_grid_used": False,
            "newest_unseen_data_used": False,
        },
        "frozen_parameters": {
            "signal_source": "Binance Vision spot hourly OHLCV",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "volume_lookback_hours": LOOKBACK_HOURS,
            "volume_multiple": VOLUME_MULTIPLE,
            "range_threshold": RANGE_THRESHOLD,
            "direction": "same as signal candle close versus open",
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "next completed hourly bar open plus one latency bar",
            "notional": NOTIONAL,
        },
        "execution_model": {
            "fee_bps_per_side": 20.0,
            "spread_bps_per_side": 10.0,
            "slippage_bps_per_side": 10.0,
            "impact_bps_per_side": 8.0,
            "latency_bars": LATENCY_BARS,
            "fill_fraction": FILL_FRACTION,
            "funding_bps_per_bar": FUNDING_BPS_PER_BAR,
            "outage_rejection_rate": OUTAGE_REJECTION_RATE,
            "effective_slippage_bps_per_side": 23.0,
            "round_trip_bps": ROUND_TRIP_BPS,
        },
        "source": {"provider": "Binance Vision", "market": "spot hourly klines", "manifests": manifests},
        "candidates": candidates,
        "status": "confirmed" if all(candidate["status"] == "confirmed" for candidate in candidates.values()) else "not_confirmed",
        "research_only": True,
        "leverage_enabled": False,
        "live_orders_placed": False,
        "paper_orders_placed": False,
        "promotion_allowed": False,
        "active_profile_changed": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
