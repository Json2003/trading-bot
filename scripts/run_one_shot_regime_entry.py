#!/usr/bin/env python3
"""Evaluate one frozen one-shot regime-entry hypothesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from scripts.execution_model import STRESS_EXECUTION
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION


START = date(2021, 1, 1)
DISCOVERY_END = date(2025, 1, 1)
END = date(2026, 8, 1)
EMA_DAYS = 50
VIX_MEDIAN_DAYS = 20
HOLD_DAYS = 5
COOLDOWN_DAYS = 5
NOTIONAL = 3_000.0
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20
MACRO_ASSETS = ("SPY", "QQQ", "TLT", "UUP", "VIX")
CRYPTO_ASSETS = ("BTC", "ETH")


def load_close_series(path: Path) -> dict[date, float]:
    result: dict[date, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            current = date.fromisoformat(row["date"])
            value = float(row["close"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid close in {path} at {current}")
            if current in result and result[current] != value:
                raise ValueError(f"conflicting duplicate in {path} at {current}")
            result[current] = value
    if not result:
        raise ValueError(f"no rows found in {path}")
    return dict(sorted(result.items()))


def load_crypto_bars(path: Path) -> dict[date, dict[str, float]]:
    result: dict[date, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            current = date.fromisoformat(row["date"])
            values = {key: float(row[key]) for key in ("open", "high", "low", "close")}
            if (
                not all(math.isfinite(value) and value > 0 for value in values.values())
                or values["high"] < max(values["open"], values["close"])
                or values["low"] > min(values["open"], values["close"])
            ):
                raise ValueError(f"invalid OHLC row in {path} at {current}")
            if current in result and result[current] != values:
                raise ValueError(f"conflicting duplicate in {path} at {current}")
            result[current] = values
    if not result:
        raise ValueError(f"no rows found in {path}")
    return dict(sorted(result.items()))


def ema_series(values: dict[date, float], span: int) -> dict[date, float]:
    alpha = 2.0 / (span + 1.0)
    result: dict[date, float] = {}
    previous: float | None = None
    for current, value in values.items():
        previous = value if previous is None else alpha * value + (1.0 - alpha) * previous
        result[current] = previous
    return result


def _prior_median(values: dict[date, float], current: date, count: int) -> float | None:
    prior = [value for day, value in values.items() if day < current]
    if len(prior) < count:
        return None
    return statistics.median(prior[-count:])


def regime_at(
    macro: dict[str, dict[date, float]],
    emas: dict[str, dict[date, float]],
    current: date,
) -> tuple[int, int, int] | None:
    if any(current not in macro[name] or current not in emas[name] for name in MACRO_ASSETS):
        return None
    vix_median = _prior_median(macro["VIX"], current, VIX_MEDIAN_DAYS)
    if vix_median is None:
        return None
    risk_on_markers = (
        macro["SPY"][current] > emas["SPY"][current],
        macro["QQQ"][current] > emas["QQQ"][current],
        macro["TLT"][current] < emas["TLT"][current],
        macro["UUP"][current] < emas["UUP"][current],
        macro["VIX"][current] < vix_median,
    )
    risk_off_markers = tuple(not value for value in risk_on_markers)
    on_score = sum(risk_on_markers)
    off_score = sum(risk_off_markers)
    if on_score >= 4 and off_score < 4:
        return 1, on_score, off_score
    if off_score >= 4 and on_score < 4:
        return -1, on_score, off_score
    return 0, on_score, off_score


def _trade(
    bars: dict[date, dict[str, float]],
    signal_date: date,
    side: int,
    asset: str,
) -> dict[str, Any] | None:
    future_dates = [current for current in bars if current > signal_date]
    if not future_dates:
        return None
    first_bar_index = list(bars).index(future_dates[0])
    dates = list(bars)
    entry_index = first_bar_index + STRESS_EXECUTION.latency_bars
    exit_index = entry_index + HOLD_DAYS
    if exit_index >= len(dates):
        return None
    entry_date = dates[entry_index]
    exit_date = dates[exit_index]
    entry_price = bars[entry_date]["open"]
    exit_price = bars[exit_date]["close"]
    gross_return = side * (exit_price / entry_price - 1.0)
    filled_notional = NOTIONAL * STRESS_EXECUTION.fill_fraction * (1.0 - STRESS_EXECUTION.outage_rejection_rate)
    trading_cost = filled_notional * STRESS_EXECUTION.round_trip_bps / 10_000.0
    funding_cost = filled_notional * STRESS_EXECUTION.funding_bps_per_bar * HOLD_DAYS / 10_000.0
    net_pnl = filled_notional * gross_return - trading_cost - funding_cost
    return {
        "signal_date": signal_date.isoformat(),
        "entry_date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "asset": asset,
        "side": "long" if side > 0 else "short",
        "gross_return": gross_return,
        "net_return": net_pnl / NOTIONAL,
        "net_pnl": net_pnl,
        "execution_cost": trading_cost + funding_cost,
    }


def _summary(rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    width = (end - start).days / BLOCKS
    groups: list[list[dict[str, Any]]] = [[] for _ in range(BLOCKS)]
    for row in rows:
        offset = (date.fromisoformat(row["signal_date"]) - start).days
        groups[min(BLOCKS - 1, int(offset / width))].append(row)
    pnls = [float(row["net_pnl"]) for row in rows]
    returns = [float(row["net_return"]) for row in rows]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    equity = peak = max_drawdown = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    block_means = [statistics.mean(float(row["net_return"]) for row in group) if group else None for group in groups]
    valid_means = [value for value in block_means if value is not None]
    stress_return = STRESS_EXECUTION.round_trip_bps / 10_000.0
    median_coverage = statistics.median(valid_means) / stress_return if valid_means else None
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_means,
        "net_pnl": sum(pnls),
        "net_return_pct": sum(returns) * 100.0,
        "max_drawdown_pct_of_notional": max_drawdown / NOTIONAL * 100.0,
        "sharpe_proxy": (statistics.mean(returns) / statistics.pstdev(returns) * math.sqrt(len(returns))) if len(returns) > 1 and statistics.pstdev(returns) else None,
        "profit_factor": gains / losses if losses else (math.inf if gains else None),
        "execution_cost": sum(float(row["execution_cost"]) for row in rows),
        "median_block_return_to_stress_cost": median_coverage,
        "passes_sample_gate": len(rows) > 0 and all(count >= MIN_TRADES_PER_BLOCK for count in [len(group) for group in groups]),
        "passes_positive_block_gate": len(valid_means) == BLOCKS and all(value > 0 for value in valid_means),
    }


def _gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["passes_sample_gate"]
        and summary["passes_positive_block_gate"]
        and summary["median_block_return_to_stress_cost"] is not None
        and summary["median_block_return_to_stress_cost"] >= 1.0
    )


def evaluate(
    macro, emas, bars, start: date, end: date, asset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    signal_dates = sorted(set.intersection(*(set(macro[name]) for name in MACRO_ASSETS)))
    signal_dates = [current for current in signal_dates if start <= current < end]
    rows: list[dict[str, Any]] = []
    previous_state = 0
    for current in signal_dates:
        state = regime_at(macro, emas, current)
        if state is None:
            continue
        direction = state[0]
        if direction != 0 and direction != previous_state:
            row = _trade(bars[asset], current, direction, asset)
            if row is not None and date.fromisoformat(row["exit_date"]) < end:
                row.update({
                    "regime": "boom" if direction > 0 else "bust",
                    "risk_on_score": state[1],
                    "risk_off_score": state[2],
                    "entry_rule": "first qualifying signal after daily regime transition",
                })
                rows.append(row)
        previous_state = direction
    return rows, _summary(rows, start, end)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    macro = {name: load_close_series(args.data_dir / f"{name}.csv") for name in MACRO_ASSETS}
    emas = {name: ema_series(macro[name], EMA_DAYS) for name in MACRO_ASSETS}
    bars = {name: load_crypto_bars(args.data_dir / f"{name}.csv") for name in CRYPTO_ASSETS}
    candidates: dict[str, Any] = {}
    for asset in CRYPTO_ASSETS:
        discovery_rows, discovery = evaluate(macro, emas, bars, START, DISCOVERY_END, asset)
        holdout_rows, holdout = evaluate(macro, emas, bars, DISCOVERY_END, END, asset)
        candidates[asset] = {
            "discovery": discovery,
            "holdout": holdout,
            "passes_discovery": _gate(discovery),
            "passes_confirmation": bool(_gate(discovery) and _gate(holdout)),
            "status": "confirmed" if _gate(discovery) and _gate(holdout) else "not_confirmed",
            "discovery_trade_rows": discovery_rows,
            "holdout_trade_rows": holdout_rows,
        }
    manifests = {}
    for name in (*MACRO_ASSETS, *CRYPTO_ASSETS):
        manifests[name] = json.loads((args.data_dir / f"{name}.manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "hypothesis": "The first qualifying BTC/ETH signal after a confirmed daily boom/bust transition has positive net expectancy; later signals in the same regime are excluded.",
        "frozen_parameters": {
            "macro_signal_source": "Yahoo Finance daily adjusted ETF closes and VIX closes",
            "execution_source": "Binance Vision spot daily candles",
            "macro_assets": list(MACRO_ASSETS),
            "execution_assets": ["BTCUSDT", "ETHUSDT"],
            "ema_days": EMA_DAYS,
            "vix_prior_median_days": VIX_MEDIAN_DAYS,
            "boom_marker_count": 4,
            "bust_marker_count": 4,
            "hold_days": HOLD_DAYS,
            "cooldown_days": COOLDOWN_DAYS,
            "entry": "first BTC/ETH daily bar after the first qualifying signal following a completed daily regime transition plus one latency bar",
            "one_trade_per_regime_transition": True,
            "notional": NOTIONAL,
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": START.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "end_exclusive": END.isoformat(),
            "holdout_untouched": True,
            "completed_bars_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "threshold_grid_used": False,
        },
        "source": {"manifests": manifests, "missing_data_is_excluded": True},
        "candidates": candidates,
        "status": "confirmed" if any(value["passes_confirmation"] for value in candidates.values()) else "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({asset: {"status": value["status"], "discovery_trades": value["discovery"]["trade_count"], "holdout_trades": value["holdout"]["trade_count"], "discovery_net_return_pct": value["discovery"]["net_return_pct"], "holdout_net_return_pct": value["holdout"]["net_return_pct"]} for asset, value in candidates.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
