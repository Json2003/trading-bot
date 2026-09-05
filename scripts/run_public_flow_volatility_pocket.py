#!/usr/bin/env python3
"""Evaluate the frozen public-flow volatility-pocket hypothesis.

The evaluator is deliberately fail-closed: incomplete, gapped, overlapping, or
short windows produce a skip report and no performance metrics.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
MINUTE = timedelta(minutes=1)
WINDOW_MINUTES = 1440
WARMUP_DURATION = timedelta(days=2)
DEVELOPMENT_DURATION = timedelta(days=30)
HOLDOUT_DURATION = timedelta(days=28)
MIN_DURATION = WARMUP_DURATION + DEVELOPMENT_DURATION + HOLDOUT_DURATION
NOTIONAL = 3000.0
ROUND_TRIP_COST = 0.0086
HOLD_MINUTES = 30
COOLDOWN_MINUTES = 30
MIN_TRADES_PER_BLOCK = 20


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    index = int(probability * (len(ordered) - 1))
    return ordered[index]


def load_rows(data_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_count = 0
    files = sorted(data_dir.rglob("completed_minute_flow.csv"))
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("completed", "").strip().lower() != "true":
                    continue
                symbol = raw.get("symbol", "").strip().upper()
                bucket = raw.get("bucket", "").strip()
                if symbol not in {"BTCUSDT", "ETHUSDT"} or not bucket:
                    continue
                try:
                    price = float(raw["last_trade_price"])
                    bid = float(raw["best_bid"])
                    ask = float(raw["best_ask"])
                    buy = float(raw["buy_notional"])
                    sell = float(raw["sell_notional"])
                    net = float(raw["net_aggressive_notional"])
                    book = float(raw["book_imbalance"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(x) for x in (price, bid, ask, buy, sell, net, book)):
                    continue
                if price <= 0 or bid <= 0 or ask <= 0 or buy + sell <= 0:
                    continue
                if bucket in grouped[symbol]:
                    duplicate_count += 1
                    continue
                grouped[symbol][bucket] = {
                    "time": parse_time(bucket),
                    "price": price,
                    "mid": (bid + ask) / 2.0,
                    "total": buy + sell,
                    "net": net,
                    "book": book,
                }
    return ({
        symbol: [grouped[symbol][key] for key in sorted(grouped[symbol])]
        for symbol in sorted(grouped)
    }, duplicate_count)


def continuity(rows: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if not rows:
        return False, None
    for previous, current in zip(rows, rows[1:]):
        if current["time"] - previous["time"] != MINUTE:
            return False, None
    return True, iso(rows[-1]["time"])


def build_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < WINDOW_MINUTES + HOLD_MINUTES + 6:
        return []
    prices = [float(row["price"]) for row in rows]
    total_values: deque[float] = deque()
    total_sorted: list[float] = []
    vol_values: deque[float] = deque()
    vol_sorted: list[float] = []
    vol5: list[float | None] = [None] * len(rows)

    for index in range(5, len(rows)):
        vol5[index] = abs(math.log(prices[index] / prices[index - 5]))

    signals: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        current_vol = vol5[index]
        if current_vol is not None:
            if len(total_values) >= WINDOW_MINUTES and len(vol_values) >= WINDOW_MINUTES:
                volume_cutoff = total_sorted[int(0.95 * (len(total_sorted) - 1))]
                volatility_cutoff = vol_sorted[int(0.90 * (len(vol_sorted) - 1))]
                flow_ratio = row["net"] / row["total"]
                direction = 0
                if (
                    row["total"] >= volume_cutoff
                    and current_vol >= volatility_cutoff
                    and flow_ratio >= 0.30
                    and row["book"] >= 0.10
                ):
                    direction = 1
                elif (
                    row["total"] >= volume_cutoff
                    and current_vol >= volatility_cutoff
                    and flow_ratio <= -0.30
                    and row["book"] <= -0.10
                ):
                    direction = -1
                if direction:
                    entry_index = index + 1
                    exit_index = entry_index + HOLD_MINUTES
                    if exit_index < len(rows):
                        signals.append(
                            {
                                "signal_time": row["time"],
                                "entry_time": rows[entry_index]["time"],
                                "exit_time": rows[exit_index]["time"],
                                "entry": rows[entry_index]["mid"],
                                "exit": rows[exit_index]["mid"],
                                "direction": direction,
                                "symbol": row.get("symbol"),
                            }
                        )
            bisect.insort(total_sorted, row["total"])
            total_values.append(row["total"])
            if len(total_values) > WINDOW_MINUTES:
                old = total_values.popleft()
                total_sorted.pop(bisect.bisect_left(total_sorted, old))
            bisect.insort(vol_sorted, current_vol)
            vol_values.append(current_vol)
            if len(vol_values) > WINDOW_MINUTES:
                old_vol = vol_values.popleft()
                vol_sorted.pop(bisect.bisect_left(vol_sorted, old_vol))
    return signals


def choose_non_overlapping(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    next_available: datetime | None = None
    for signal in sorted(signals, key=lambda item: (item["entry_time"], item["symbol"] or "")):
        if next_available is not None and signal["entry_time"] < next_available:
            continue
        signal["gross_return"] = signal["direction"] * (
            signal["exit"] / signal["entry"] - 1.0
        )
        signal["net_return"] = signal["gross_return"] - ROUND_TRIP_COST
        signal["net_pnl"] = signal["net_return"] * NOTIONAL
        chosen.append(signal)
        next_available = signal["exit_time"] + timedelta(minutes=COOLDOWN_MINUTES)
    return chosen


def metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    net_pnl = sum(float(trade["net_pnl"]) for trade in trades)
    net_returns = [float(trade["net_return"]) for trade in trades]
    wins = [value for value in net_returns if value > 0]
    losses = [value for value in net_returns if value < 0]
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        equity += float(trade["net_pnl"])
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    if len(net_returns) >= 2:
        mean = sum(net_returns) / len(net_returns)
        variance = sum((value - mean) ** 2 for value in net_returns) / (len(net_returns) - 1)
        sharpe = mean / math.sqrt(variance) * math.sqrt(len(net_returns)) if variance else None
    else:
        sharpe = None
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    return {
        "net_pnl": round(net_pnl, 2),
        "net_return_pct_fixed_notional": round(net_pnl / NOTIONAL * 100.0, 4),
        "max_drawdown_pct_fixed_notional": round(max_drawdown / NOTIONAL * 100.0, 4),
        "sharpe_proxy": round(sharpe, 4) if sharpe is not None else None,
        "profit_factor": round(gross_wins / gross_losses, 4) if gross_losses else (None if not wins else "undefined"),
        "trade_count": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 2) if trades else None,
        "execution_costs": round(len(trades) * NOTIONAL * ROUND_TRIP_COST, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-evaluation", action="store_true",
                        help="Return nonzero when archive validation skips evaluation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_symbol, duplicate_count = load_rows(args.data_dir)
    continuity_info: dict[str, Any] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        rows = rows_by_symbol.get(symbol, [])
        continuous, through = continuity(rows)
        continuity_info[symbol] = {
            "rows": len(rows),
            "continuous": continuous,
            "data_through": through,
            "data_from": iso(rows[0]["time"]) if rows else None,
        }

    report: dict[str, Any] = {
        "hypothesis": "public_flow_confirmed_volatility_pocket_continuation",
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "promotion_allowed": False,
        "parameters": {
            "volume_prior_minutes": WINDOW_MINUTES,
            "volume_percentile": 0.95,
            "volatility_prior_minutes": WINDOW_MINUTES,
            "volatility_percentile": 0.90,
            "flow_imbalance": 0.30,
            "book_imbalance": 0.10,
            "latency_minutes": 1,
            "hold_minutes": HOLD_MINUTES,
            "cooldown_minutes": COOLDOWN_MINUTES,
            "round_trip_cost_bps": 86,
            "fixed_trade_notional": NOTIONAL,
        },
        "continuity": continuity_info,
        "duplicate_row_count": duplicate_count,
        "status": "skip",
        "confirmed": False,
        "reason": None,
        "segments": {},
        "blocks": [],
    }

    starts = [
        parse_time(info["data_from"])
        for info in continuity_info.values()
        if info["data_from"]
    ]
    ends = [
        parse_time(info["data_through"])
        for info in continuity_info.values()
        if info["data_through"]
    ]
    if len(starts) != 2 or len(ends) != 2:
        report["reason"] = "missing BTCUSDT or ETHUSDT completed public-flow data"
    elif duplicate_count:
        report["reason"] = "overlapping or duplicate minute rows detected"
    elif any(not info["continuous"] for info in continuity_info.values()):
        report["reason"] = "gap or malformed timestamp sequence"
    else:
        start = max(starts)
        end = min(ends)
        if end - start + MINUTE < MIN_DURATION:
            report["reason"] = "continuous public-flow window is shorter than 60 days"
        else:
            evaluation_start = start + WARMUP_DURATION
            development_end = evaluation_start + DEVELOPMENT_DURATION
            holdout_end = development_end + HOLDOUT_DURATION
            all_signals: list[dict[str, Any]] = []
            for symbol, rows in rows_by_symbol.items():
                usable = [row for row in rows if start <= row["time"] < holdout_end]
                for row in usable:
                    row["symbol"] = symbol
                all_signals.extend(build_signals(usable))
            # Exclude boundary-crossing outcomes before portfolio selection.
            # Development P&L must never use a confirmation-minute price.
            eligible_signals = [
                signal for signal in all_signals
                if evaluation_start <= signal["signal_time"]
                and signal["exit_time"] < holdout_end
                and not (
                    signal["signal_time"] < development_end
                    and signal["exit_time"] >= development_end
                )
            ]
            trades = choose_non_overlapping(eligible_signals)
            for trade in trades:
                if trade["entry_time"] < evaluation_start or trade["exit_time"] > holdout_end:
                    trade["block"] = -1
                    continue
                if trade["entry_time"] < development_end:
                    fraction = (
                        (trade["entry_time"] - evaluation_start).total_seconds()
                        / DEVELOPMENT_DURATION.total_seconds()
                    )
                    trade["block"] = min(3, int(fraction * 4.0))
                else:
                    fraction = (
                        (trade["entry_time"] - development_end).total_seconds()
                        / HOLDOUT_DURATION.total_seconds()
                    )
                    trade["block"] = 4 + min(1, int(fraction * 2.0))
            trades = [trade for trade in trades if trade["block"] >= 0]
            development = [trade for trade in trades if trade["block"] < 4]
            holdout = [trade for trade in trades if trade["block"] >= 4]
            duration = holdout_end - evaluation_start
            report["status"] = "evaluated"
            report["data_from"] = iso(start)
            report["evaluation_from"] = iso(evaluation_start)
            report["data_through"] = iso(holdout_end - MINUTE)
            report["source_duration_days"] = round(
                (end - start + MINUTE).total_seconds() / 86400.0, 4
            )
            report["duration_days"] = round(duration.total_seconds() / 86400.0, 4)
            report["segments"] = {
                "development": metrics(development),
                "untouched_confirmation": metrics(holdout),
            }
            report["blocks"] = [
                {"block": block, **metrics([trade for trade in trades if trade["block"] == block])}
                for block in range(6)
            ]
            report["confirmed"] = (
                all(block["trade_count"] >= MIN_TRADES_PER_BLOCK and block["net_return_pct_fixed_notional"] > 0 for block in report["blocks"])
                and report["segments"]["development"]["net_return_pct_fixed_notional"] > 0
                and report["segments"]["untouched_confirmation"]["net_return_pct_fixed_notional"] > 0
            )
            report["reason"] = "passed all frozen block and confirmation gates" if report["confirmed"] else "did not pass frozen confirmation gates"

    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.require_evaluation and report["status"] != "evaluated" else 0


if __name__ == "__main__":
    raise SystemExit(main())
