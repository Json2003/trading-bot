#!/usr/bin/env python3
"""Single frozen OHLCV shock-continuation experiment.

Research-only. One rule, one six-hour horizon, fixed dates, no tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.execution_model import STRESS_EXECUTION
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from execution_model import STRESS_EXECUTION
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 1, tzinfo=timezone.utc)
DISCOVERY_END = datetime(2025, 4, 1, tzinfo=timezone.utc)
HORIZON_HOURS = 6
COOLDOWN_HOURS = 12
NOTIONAL_PER_ASSET = 3_000.0
SHOCK_Z = -2.0
VOLUME_MULTIPLE = 1.5
RANGE_PERCENTILE_MIN = 0.75
EMA_PERIOD = 200
RETURN_LOOKBACK = 720
VOLUME_LOOKBACK = 24
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20
STRESS_MULTIPLE_GATE = 1.0


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def ema(values: list[float], period: int) -> list[float]:
    out = [math.nan] * len(values)
    if len(values) < period:
        return out
    value = mean(values[:period])
    out[period - 1] = value
    alpha = 2.0 / (period + 1.0)
    for i in range(period, len(values)):
        value = alpha * values[i] + (1.0 - alpha) * value
        out[i] = value
    return out


def prior_std(values: list[float], i: int, window: int) -> float:
    sample = values[max(0, i - window):i]
    if len(sample) < 120:
        return math.nan
    avg = mean(sample)
    return math.sqrt(mean([(x - avg) ** 2 for x in sample]))


def features(bars: list) -> dict[str, list[float]]:
    close = [float(x.close) for x in bars]
    volume = [float(x.volume) for x in bars]
    returns = [math.nan] * len(close)
    ranges = [math.nan] * len(close)
    for i in range(1, len(close)):
        returns[i] = math.log(close[i] / close[i - 1])
        ranges[i] = (float(bars[i].high) - float(bars[i].low)) / close[i]
    range_percentile = [math.nan] * len(close)
    for i in range(VOLUME_LOOKBACK, len(close)):
        lows = [ranges[j] for j in range(i - VOLUME_LOOKBACK, i) if finite(ranges[j])]
        range_percentile[i] = (
            sum(ranges[i] >= value for value in lows) / len(lows)
            if lows and finite(ranges[i]) else math.nan
        )
    return {
        "close": close,
        "volume": volume,
        "returns": returns,
        "ema": ema(close, EMA_PERIOD),
        "range_percentile": range_percentile,
    }


def empty_diagnostics() -> dict[str, int]:
    return {
        "observations": 0,
        "finite_features": 0,
        "nonfinite_features": 0,
        "zero_or_invalid_volatility": 0,
        "shock_pass": 0,
        "volume_pass": 0,
        "range_percentile_pass": 0,
        "trend_pass": 0,
        "signals": 0,
    }


def is_signal(
    i: int,
    data: dict[str, list[float]],
    diagnostics: dict[str, int],
) -> bool:
    if i < max(EMA_PERIOD, RETURN_LOOKBACK, VOLUME_LOOKBACK):
        return False
    diagnostics["observations"] += 1
    prior = [x for x in data["returns"][max(0, i - RETURN_LOOKBACK):i] if finite(x)]
    if len(prior) < 120:
        diagnostics["nonfinite_features"] += 1
        return False
    prior_mean = mean(prior)
    variance = mean([(x - prior_mean) ** 2 for x in prior])
    if not finite(variance) or variance <= 0:
        diagnostics["zero_or_invalid_volatility"] += 1
        return False
    z = (data["returns"][i] - prior_mean) / math.sqrt(variance)
    prior_volume = data["volume"][i - VOLUME_LOOKBACK:i]
    features_are_finite = finite(
        data["returns"][i],
        data["ema"][i],
        data["range_percentile"][i],
        data["volume"][i],
        *prior_volume,
    )
    if not features_are_finite:
        diagnostics["nonfinite_features"] += 1
        return False
    diagnostics["finite_features"] += 1
    if z > SHOCK_Z:
        return False
    diagnostics["shock_pass"] += 1
    if data["volume"][i] < VOLUME_MULTIPLE * median(prior_volume):
        return False
    diagnostics["volume_pass"] += 1
    if data["range_percentile"][i] < RANGE_PERCENTILE_MIN:
        return False
    diagnostics["range_percentile_pass"] += 1
    if data["close"][i] <= data["ema"][i]:
        return False
    diagnostics["trend_pass"] += 1
    diagnostics["signals"] += 1
    return True


def trade(pair: list, data: dict[str, list[float]], symbol: str, i: int) -> dict | None:
    entry_i = i + 1 + STRESS_EXECUTION.latency_bars
    exit_i = entry_i + HORIZON_HOURS
    if exit_i >= len(pair):
        return None
    asset = pair[entry_i].btc if symbol == "BTC" else pair[entry_i].eth
    exit_asset = pair[exit_i].btc if symbol == "BTC" else pair[exit_i].eth
    entry = float(asset.open)
    exit_price = float(exit_asset.close)
    if entry <= 0:
        return None
    gross = exit_price / entry - 1.0
    filled = NOTIONAL_PER_ASSET * STRESS_EXECUTION.fill_fraction * (
        1.0 - STRESS_EXECUTION.outage_rejection_rate
    )
    trading_cost = filled * STRESS_EXECUTION.round_trip_bps / 10_000.0
    funding_cost = (
        filled * STRESS_EXECUTION.funding_bps_per_bar * HORIZON_HOURS / 10_000.0
    )
    net_pnl = filled * gross - trading_cost - funding_cost
    return {
        "index": i,
        "timestamp": pair[i].btc.timestamp.isoformat(),
        "symbol": symbol,
        "gross_return": gross,
        "net_pnl": net_pnl,
        "net_return_on_notional": net_pnl / NOTIONAL_PER_ASSET,
        "execution_cost": trading_cost + funding_cost,
    }


def continuation_confirmation(pair: list, symbol: str, i: int) -> bool:
    """Require the next completed candle to confirm downside continuation."""
    confirm_i = i + 1
    if confirm_i >= len(pair):
        return False
    shock = pair[i].btc if symbol == "BTC" else pair[i].eth
    confirm = pair[confirm_i].btc if symbol == "BTC" else pair[confirm_i].eth
    midpoint = (float(shock.open) + float(shock.close)) / 2.0
    return float(confirm.close) < float(confirm.open) and float(confirm.close) <= midpoint


def evaluate_segment(
    pair: list, features_by_symbol: dict[str, dict[str, list[float]]],
    start: int, end: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rows = []
    diagnostics = {
        symbol: empty_diagnostics() for symbol in ("BTC", "ETH")
    }
    last_signal = {"BTC": -10**9, "ETH": -10**9}
    for i in range(max(start, RETURN_LOOKBACK), end - HORIZON_HOURS):
        for symbol in ("BTC", "ETH"):
            if i - last_signal[symbol] < COOLDOWN_HOURS:
                continue
            if is_signal(
                i,
                features_by_symbol[symbol],
                diagnostics[symbol],
            ) and continuation_confirmation(pair, symbol, i):
                row = trade(pair, features_by_symbol[symbol], symbol, i)
                if row is not None:
                    rows.append(row)
                    last_signal[symbol] = i
    return rows, diagnostics


def summarize(rows: list[dict], start: int, end: int) -> dict:
    width = (end - start) / BLOCKS
    groups = [[] for _ in range(BLOCKS)]
    for row in rows:
        block = min(BLOCKS - 1, int((row["index"] - start) / width))
        groups[block].append(row)
    pnl = [row["net_pnl"] for row in rows]
    returns = [row["net_return_on_notional"] for row in rows]
    gains = sum(x for x in pnl if x > 0)
    losses = abs(sum(x for x in pnl if x < 0))
    block_means = [mean([x["net_return_on_notional"] for x in group]) if group else None for group in groups]
    valid = [x for x in block_means if x is not None]
    median_block = median(valid) if valid else math.nan
    stress_cost = STRESS_EXECUTION.round_trip_bps / 10_000.0
    return {
        "trade_count": len(rows),
        "block_trade_counts": [len(group) for group in groups],
        "block_mean_net_returns": block_means,
        "net_pnl": sum(pnl),
        "net_return_pct": sum(returns) * 100.0,
        "median_net_trade_return": median(returns) if returns else None,
        "profit_factor": gains / losses if losses else None,
        "execution_cost": sum(row["execution_cost"] for row in rows),
        "median_block_return_to_stress_cost": (
            median_block / stress_cost if finite(median_block) else None
        ),
        "passes_sample_gate": (
            len(rows) >= BLOCKS * MIN_TRADES_PER_BLOCK
            and all(len(group) >= MIN_TRADES_PER_BLOCK for group in groups)
        ),
        "passes_positive_block_gate": (
            len(valid) == BLOCKS and all(value > 0 for value in valid)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_pair = align_pair(load_bars(args.btc_path), load_bars(args.eth_path))
    pair = [x for x in raw_pair if START <= x.btc.timestamp < END]
    if not pair or pair[-1].btc.timestamp < DISCOVERY_END:
        raise ValueError("fixed 2023-01-01 through 2026-07-31 window is incomplete")
    features_by_symbol = {
        "BTC": features([x.btc for x in pair]),
        "ETH": features([x.eth for x in pair]),
    }
    discovery_end = next(
        i for i, item in enumerate(pair) if item.btc.timestamp >= DISCOVERY_END
    )
    discovery_rows, discovery_diagnostics = evaluate_segment(
        pair, features_by_symbol, 0, discovery_end
    )
    holdout_rows, holdout_diagnostics = evaluate_segment(
        pair, features_by_symbol, discovery_end, len(pair)
    )
    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "hypothesis": "high-volume downside shock continues after next-candle confirmation over the next six hours",
        "frozen_parameters": {
            "assets": ["BTCUSDT", "ETHUSDT"],
            "horizon_hours": HORIZON_HOURS,
            "shock_z": SHOCK_Z,
            "volume_multiple": VOLUME_MULTIPLE,
            "range_percentile": RANGE_PERCENTILE_MIN,
            "ema_filter": "none",
            "return_lookback_hours": RETURN_LOOKBACK,
            "volume_lookback_hours": VOLUME_LOOKBACK,
            "cooldown_hours": COOLDOWN_HOURS,
            "notional_per_asset": NOTIONAL_PER_ASSET,
            "position_selection": "both_assets_independently; no_leader_selection",
        },
        "execution_model": STRESS_EXECUTION.as_dict(),
        "window": {
            "start": START.isoformat(),
            "end_exclusive": END.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "holdout_untouched": True,
            "completed_candles_only": True,
        },
        "discovery": summarize(discovery_rows, 0, discovery_end),
        "holdout": summarize(holdout_rows, discovery_end, len(pair)),
        "diagnostics": {
            "discovery": discovery_diagnostics,
            "holdout": holdout_diagnostics,
            "note": "Counters are observational only; frozen signal predicates are unchanged.",
        },
    }
    discovery = report["discovery"]
    holdout = report["holdout"]
    report["passes_discovery"] = (
        discovery["passes_sample_gate"]
        and discovery["passes_positive_block_gate"]
        and (discovery["median_block_return_to_stress_cost"] or 0.0) >= STRESS_MULTIPLE_GATE
    )
    report["passes_confirmation"] = (
        report["passes_discovery"]
        and holdout["passes_sample_gate"]
        and holdout["passes_positive_block_gate"]
        and (holdout["median_block_return_to_stress_cost"] or 0.0) >= STRESS_MULTIPLE_GATE
    )
    report["status"] = "confirmed" if report["passes_confirmation"] else "not_confirmed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "passes_discovery": report["passes_discovery"],
        "passes_confirmation": report["passes_confirmation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
