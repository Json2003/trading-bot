#!/usr/bin/env python3
"""Evaluate a frozen top-trader positioning plus high-volume continuation rule.

This is a research-only evaluator.  Binance's top-trader series is an
anonymous cohort proxy, not a list of individual people's trades.
"""

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
    from scripts.run_momentum_volatility_research import Bar, load_bars
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import Bar, load_bars

NOTIONAL = 3_000.0
PRICE_MOVE_THRESHOLD = 0.005
VOLUME_LOOKBACK_HOURS = 24
VOLUME_MULTIPLE = 1.5
TOP_POSITION_LONG_THRESHOLD = 0.60
TOP_ACCOUNT_LONG_THRESHOLD = 0.55
HOLD_HOURS = 6
COOLDOWN_HOURS = 12
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 5
UTC = timezone.utc


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def load_positioning(path: Path) -> dict[datetime, dict[str, float]]:
    result: dict[datetime, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _utc(row["timestamp"])
            values = {
                key: float(row[key])
                for key in (
                    "account_long",
                    "account_short",
                    "position_long",
                    "position_short",
                    "account_long_short_ratio",
                    "position_long_short_ratio",
                )
            }
            if not all(math.isfinite(value) for value in values.values()):
                continue
            if not (
                0 <= values["account_long"] <= 1
                and 0 <= values["account_short"] <= 1
                and 0 <= values["position_long"] <= 1
                and 0 <= values["position_short"] <= 1
                and values["account_long_short_ratio"] > 0
                and values["position_long_short_ratio"] > 0
            ):
                continue
            result[timestamp] = values
    return result



def _coverage(
    positioning: dict[datetime, dict[str, float]],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Require one aligned top-trader observation for every evaluation hour."""
    expected_hours = max(0, int((end - start).total_seconds() // 3600))
    missing = []
    cursor = start
    for _ in range(expected_hours):
        if cursor not in positioning:
            missing.append(cursor.isoformat())
        cursor += timedelta(hours=1)
    timestamps = sorted(positioning)
    return {
        "required_start": start.isoformat(),
        "required_end_exclusive": end.isoformat(),
        "expected_hour_count": expected_hours,
        "observed_row_count": len(positioning),
        "missing_hour_count": len(missing),
        "missing_hours_sample": missing[:10],
        "first_timestamp": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
        "complete": bool(timestamps) and not missing,
    }


def _signal(
    bars: list[Bar],
    positioning: dict[datetime, dict[str, float]],
    index: int,
) -> tuple[int, float, float, dict[str, float]] | None:
    if index < VOLUME_LOOKBACK_HOURS or index == 0:
        return None
    bar = bars[index]
    previous = bars[index - 1]
    if previous.timestamp != bar.timestamp - timedelta(hours=1):
        return None
    cohort = positioning.get(bar.timestamp)
    if cohort is None:
        return None
    prior_volumes = [float(item.volume) for item in bars[index - VOLUME_LOOKBACK_HOURS : index]]
    if not prior_volumes or any(
        not math.isfinite(value) or value < 0 for value in prior_volumes
    ):
        return None
    baseline_volume = statistics.median(prior_volumes)
    if not math.isfinite(baseline_volume) or baseline_volume <= 0:
        return None
    price_move = bar.close / previous.close - 1.0
    volume_ratio = float(bar.volume) / baseline_volume
    if (
        not math.isfinite(price_move)
        or not math.isfinite(volume_ratio)
        or abs(price_move) < PRICE_MOVE_THRESHOLD
        or volume_ratio < VOLUME_MULTIPLE
    ):
        return None

    long_crowded = (
        cohort["position_long"] >= TOP_POSITION_LONG_THRESHOLD
        and cohort["account_long"] >= TOP_ACCOUNT_LONG_THRESHOLD
    )
    short_crowded = (
        cohort["position_short"] >= TOP_POSITION_LONG_THRESHOLD
        and cohort["account_short"] >= TOP_ACCOUNT_LONG_THRESHOLD
    )
    # Follow the crowded top-trader side only when the completed high-volume
    # price impulse agrees with it.  Entry waits for the next bar plus latency.
    if price_move > 0 and long_crowded:
        side = 1
    elif price_move < 0 and short_crowded:
        side = -1
    else:
        return None
    return side, price_move, volume_ratio, cohort


def _trade(
    bars: list[Bar],
    signal_index: int,
    side: int,
    symbol: str,
) -> dict[str, Any] | None:
    entry_index = signal_index + 1 + STRESS_EXECUTION.latency_bars
    exit_index = entry_index + HOLD_HOURS
    if exit_index >= len(bars):
        return None
    entry = float(bars[entry_index].open)
    exit_price = float(bars[exit_index].close)
    if not (
        math.isfinite(entry)
        and math.isfinite(exit_price)
        and entry > 0
        and exit_price > 0
    ):
        return None
    gross_return = side * (exit_price / entry - 1.0)
    filled_notional = (
        NOTIONAL
        * STRESS_EXECUTION.fill_fraction
        * (1.0 - STRESS_EXECUTION.outage_rejection_rate)
    )
    trading_cost = (
        filled_notional * STRESS_EXECUTION.round_trip_bps / 10_000.0
    )
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


def _summary(
    rows: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    pnls = [float(row["net_pnl"]) for row in rows]
    returns = [float(row["net_return"]) for row in rows]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    equity = peak = max_drawdown = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    width = (end - start).total_seconds() / BLOCKS
    block_counts = [0] * BLOCKS
    block_returns: list[list[float]] = [[] for _ in range(BLOCKS)]
    start_epoch = start.timestamp()
    for row in rows:
        timestamp = _utc(row["signal_timestamp"]).timestamp()
        block = min(BLOCKS - 1, int((timestamp - start_epoch) / width))
        block_counts[block] += 1
        block_returns[block].append(float(row["net_return"]))
    block_means = [
        statistics.mean(values) if values else None for values in block_returns
    ]
    mean_return = statistics.mean(returns) if returns else None
    volatility = statistics.pstdev(returns) if len(returns) > 1 else None
    valid_means = [value for value in block_means if value is not None]
    return {
        "trade_count": len(rows),
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
        "block_trade_counts": block_counts,
        "block_mean_net_returns": block_means,
        "positive_block_count": sum(
            1 for value in block_means if value is not None and value > 0
        ),
        "median_block_return_to_stress_cost": (
            statistics.median(valid_means)
            / (STRESS_EXECUTION.round_trip_bps / 10_000.0)
            if valid_means
            else None
        ),
        "passes_sample_gate": all(
            count >= MIN_TRADES_PER_BLOCK for count in block_counts
        ),
        "passes_positive_block_gate": (
            len(valid_means) == BLOCKS and all(value > 0 for value in valid_means)
        ),
    }


def _evaluate(
    bars: list[Bar],
    positioning: dict[datetime, dict[str, float]],
    start: datetime,
    end: datetime,
    symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = next(i for i, bar in enumerate(bars) if bar.timestamp >= start)
    end_index = next(
        (i for i, bar in enumerate(bars) if bar.timestamp >= end), len(bars)
    )
    rows: list[dict[str, Any]] = []
    last_signal_index = -10**9
    for index in range(first, max(first, end_index - HOLD_HOURS - 2)):
        signal = _signal(bars, positioning, index)
        if signal is None:
            continue
        side, price_move, volume_ratio, cohort = signal
        if index - last_signal_index < COOLDOWN_HOURS:
            continue
        row = _trade(bars, index, side, symbol)
        if row is None:
            continue
        row.update(
            {
                "price_move_1h": price_move,
                "volume_ratio_24h_median": volume_ratio,
                "top_account_long": cohort["account_long"],
                "top_account_short": cohort["account_short"],
                "top_position_long": cohort["position_long"],
                "top_position_short": cohort["position_short"],
            }
        )
        rows.append(row)
        last_signal_index = index
    rows.sort(key=lambda row: row["signal_timestamp"])
    return rows, _summary(rows, start, end)


def _skip_report(
    output: Path,
    start: datetime,
    development_end: datetime,
    end: datetime,
    reason: str,
    manifests: dict[str, Any],
    status: str = "skipped_missing_top_trader_history",
) -> None:
    null_summary = {
        "trade_count": 0,
        "net_pnl": None,
        "net_return_pct": None,
        "max_drawdown_pct_of_notional": None,
        "sharpe_proxy": None,
        "profit_factor": None,
        "execution_cost": None,
        "block_trade_counts": [0] * BLOCKS,
        "block_mean_net_returns": [None] * BLOCKS,
        "positive_block_count": 0,
        "passes_sample_gate": False,
        "passes_positive_block_gate": False,
        "median_block_return_to_stress_cost": None,
    }
    report = _report_base(start, development_end, end, manifests)
    report.update(
        {
            "status": status,
            "skip_reason": reason,
            "candidates": {
                "BTC": {
                    "discovery": null_summary,
                    "holdout": null_summary,
                    "status": "skipped",
                },
                "ETH": {
                    "discovery": null_summary,
                    "holdout": null_summary,
                    "status": "skipped",
                },
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _report_base(
    start: datetime,
    development_end: datetime,
    end: datetime,
    manifests: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": (
            "When anonymous top-trader positioning and a completed high-volume "
            "price impulse agree, the impulse continues for six hours."
        ),
        "frozen_parameters": {
            "positioning_source": (
                "Binance USD-M top 20% of users by margin balance; "
                "account and position ratios"
            ),
            "execution_source": "Binance spot hourly candles",
            "assets": ["BTCUSDT", "ETHUSDT"],
            "top_position_long_or_short_threshold": TOP_POSITION_LONG_THRESHOLD,
            "top_account_long_or_short_threshold": TOP_ACCOUNT_LONG_THRESHOLD,
            "price_move_threshold": PRICE_MOVE_THRESHOLD,
            "volume_lookback_hours": VOLUME_LOOKBACK_HOURS,
            "volume_multiple": VOLUME_MULTIPLE,
            "direction": "follow the crowded cohort when the price impulse agrees",
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "entry": "signal-hour close, next-bar open plus one latency bar",
            "notional": NOTIONAL,
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": start.isoformat(),
            "development_end_exclusive": development_end.isoformat(),
            "end_exclusive": end.isoformat(),
            "development_holdout_split": True,
            "holdout_untouched": True,
            "completed_candles_only": True,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "threshold_grid_used": False,
            "retuned_after_holdout": False,
            "one_year_history_available": False,
            "newest_unseen_data_used": True,
            "source_coverage_required": "one aligned completed top-trader observation for every evaluation hour",
            "evaluation_days": int((end - start).total_seconds() // 86400),
            "development_days": int((development_end - start).total_seconds() // 86400),
            "holdout_days": int((end - development_end).total_seconds() // 86400),
        },
        "source": {
            "provider": "Binance",
            "top_trader_history_limit": "latest 30 days only",
            "manifests": manifests,
            "individual_trader_identity_available": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--positioning-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--development-end", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = _utc(args.start)
    development_end = _utc(args.development_end)
    end = _utc(args.end)
    manifests = {}
    for asset, symbol in {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}.items():
        manifest_path = args.positioning_dir / f"{symbol}_1h.manifest.json"
        manifests[asset] = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"available": False, "reason": "manifest missing"}
        )
    if not all(manifest.get("available") for manifest in manifests.values()):
        reason = "; ".join(
            f"{asset}: {manifest.get('reason', 'unavailable')}"
            for asset, manifest in manifests.items()
            if not manifest.get("available")
        )
        _skip_report(args.output, start, development_end, end, reason, manifests)
        print(json.dumps({"status": "skipped", "reason": reason}, indent=2))
        return 0

    positioning = {
        asset: load_positioning(
            args.positioning_dir / f"{symbol}_1h.csv"
        )
        for asset, symbol in {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}.items()
    }
    coverage = {}
    for asset in ("BTC", "ETH"):
        coverage[asset] = _coverage(positioning[asset], start, end)
        manifests[asset]["evaluation_coverage"] = coverage[asset]
    incomplete = [
        f"{asset}: top-trader archive does not cover every evaluation hour "
        f"({coverage[asset]['missing_hour_count']} missing)"
        for asset in ("BTC", "ETH")
        if not coverage[asset]["complete"]
    ]
    if incomplete:
        reason = "; ".join(incomplete)
        _skip_report(
            args.output,
            start,
            development_end,
            end,
            reason,
            manifests,
            status="skipped_incomplete_top_trader_archive",
        )
        print(json.dumps({"status": "skipped", "reason": reason}, indent=2))
        return 0
    bars_by_asset = {
        asset: load_bars(path)
        for asset, path in {"BTC": args.btc_path, "ETH": args.eth_path}.items()
    }
    candidates: dict[str, Any] = {}
    for asset in ("BTC", "ETH"):
        discovery_rows, discovery = _evaluate(
            bars_by_asset[asset],
            positioning[asset],
            start,
            development_end,
            asset,
        )
        holdout_rows, holdout = _evaluate(
            bars_by_asset[asset],
            positioning[asset],
            development_end,
            end,
            asset,
        )
        discovery_gate = bool(
            discovery["passes_sample_gate"]
            and discovery["passes_positive_block_gate"]
            and (
                discovery["median_block_return_to_stress_cost"] is not None
                and discovery["median_block_return_to_stress_cost"] >= 1.0
            )
        )
        confirmation_gate = bool(
            discovery_gate
            and holdout["passes_sample_gate"]
            and holdout["passes_positive_block_gate"]
            and (
                holdout["median_block_return_to_stress_cost"] is not None
                and holdout["median_block_return_to_stress_cost"] >= 1.0
            )
        )
        candidates[asset] = {
            "discovery": discovery,
            "holdout": holdout,
            "passes_discovery": discovery_gate,
            "passes_confirmation": confirmation_gate,
            "status": "confirmed" if confirmation_gate else "not_confirmed",
            "discovery_trade_rows": discovery_rows,
            "holdout_trade_rows": holdout_rows,
        }

    report = _report_base(start, development_end, end, manifests)
    report["candidates"] = candidates
    report["status"] = (
        "confirmed"
        if any(candidate["passes_confirmation"] for candidate in candidates.values())
        else "not_confirmed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                asset: {
                    "status": candidate["status"],
                    "discovery_trades": candidate["discovery"]["trade_count"],
                    "holdout_trades": candidate["holdout"]["trade_count"],
                    "discovery_net_return_pct": candidate["discovery"][
                        "net_return_pct"
                    ],
                    "holdout_net_return_pct": candidate["holdout"]["net_return_pct"],
                }
                for asset, candidate in candidates.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
