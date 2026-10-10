#!/usr/bin/env python3
"""Bounded discovery-only screen for one-shot regime-entry parameters."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import date
from pathlib import Path
from typing import Any

from scripts.run_one_shot_regime_entry import _summary, _trade, ema_series, load_close_series, load_crypto_bars

MACRO_ASSETS = ("SPY", "QQQ", "TLT", "UUP", "VIX")
CRYPTO_ASSETS = ("BTC", "ETH")
START = date(2021, 1, 1)
DISCOVERY_END = date(2025, 1, 1)
END = date(2026, 8, 1)
PARAM_GRID = {
    "ema_days": (20, 50, 100),
    "vix_median_days": (10, 20, 40),
    "marker_count": (4, 5),
}
MIN_DISCOVERY_TRADES = 60

def prior_median(values, current, count):
    prior = [value for day, value in values.items() if day < current]
    return statistics.median(prior[-count:]) if len(prior) >= count else None

def regime_at_params(macro, emas, current, vix_days, marker_count):
    if any(current not in macro[n] or current not in emas[n] for n in MACRO_ASSETS):
        return None
    median = prior_median(macro["VIX"], current, vix_days)
    if median is None:
        return None
    on = (
        macro["SPY"][current] > emas["SPY"][current],
        macro["QQQ"][current] > emas["QQQ"][current],
        macro["TLT"][current] < emas["TLT"][current],
        macro["UUP"][current] < emas["UUP"][current],
        macro["VIX"][current] < median,
    )
    risk_on = sum(on)
    risk_off = 5 - risk_on
    if risk_on >= marker_count and risk_off < marker_count:
        return 1, risk_on, risk_off
    if risk_off >= marker_count and risk_on < marker_count:
        return -1, risk_on, risk_off
    return 0, risk_on, risk_off

def evaluate(macro, bars, start, end, ema_days, vix_days, marker_count, asset):
    emas = {name: ema_series(macro[name], ema_days) for name in MACRO_ASSETS}
    dates = sorted(set.intersection(*(set(macro[n]) for n in MACRO_ASSETS)))
    previous_state = 0
    rows = []
    for current in (d for d in dates if start <= d < end):
        state = regime_at_params(macro, emas, current, vix_days, marker_count)
        if state is None:
            continue
        direction = state[0]
        if direction and direction != previous_state:
            row = _trade(bars[asset], current, direction, asset)
            if row is not None and date.fromisoformat(row["exit_date"]) < end:
                row.update({"regime": "boom" if direction > 0 else "bust",
                            "risk_on_score": state[1], "risk_off_score": state[2]})
                rows.append(row)
        previous_state = direction
    return rows, _summary(rows, start, end)

def discovery_score(summary):
    means = [x for x in summary["block_mean_net_returns"] if x is not None]
    positive_blocks = sum(x > 0 for x in means)
    median = statistics.median(means) if means else -math.inf
    return (positive_blocks, median, summary["net_pnl"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    macro = {n: load_close_series(args.data_dir / f"{n}.csv") for n in MACRO_ASSETS}
    bars = {n: load_crypto_bars(args.data_dir / f"{n}.csv") for n in CRYPTO_ASSETS}

    candidates = []
    for ema_days in PARAM_GRID["ema_days"]:
        for vix_days in PARAM_GRID["vix_median_days"]:
            for marker_count in PARAM_GRID["marker_count"]:
                asset_results = {}
                for asset in CRYPTO_ASSETS:
                    d_rows, d_summary = evaluate(macro, bars, START, DISCOVERY_END, ema_days, vix_days, marker_count, asset)
                    h_rows, h_summary = evaluate(macro, bars, DISCOVERY_END, END, ema_days, vix_days, marker_count, asset)
                    asset_results[asset] = {
                        "discovery": d_summary,
                        "holdout_diagnostic": h_summary,
                        "discovery_trade_rows": d_rows,
                        "holdout_trade_rows": h_rows,
                    }
                combined_discovery_rows = [r for a in CRYPTO_ASSETS for r in asset_results[a]["discovery_trade_rows"]]
                combined_summary = _summary(combined_discovery_rows, START, DISCOVERY_END)
                candidates.append({
                    "parameters": {"ema_days": ema_days, "vix_median_days": vix_days, "marker_count": marker_count},
                    "combined_discovery": combined_summary,
                    "assets": asset_results,
                    "eligible_for_selection": combined_summary["trade_count"] >= MIN_DISCOVERY_TRADES,
                })

    eligible = [c for c in candidates if c["eligible_for_selection"]]
    selected = max(eligible, key=lambda c: discovery_score(c["combined_discovery"])) if eligible else None
    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "promotion_allowed": False,
        "selection_rule": "Among candidates with >=60 combined discovery trades, maximize positive discovery block count, then median block net return, then net P&L; no holdout values used.",
        "parameter_grid": PARAM_GRID,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected_parameters": selected["parameters"] if selected else None,
        "selected_discovery": selected["combined_discovery"] if selected else None,
        "selected_asset_results": selected["assets"] if selected else None,
        "all_candidates": candidates,
        "holdout_used_for_selection": False,
        "status": "not_confirmed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected_parameters": report["selected_parameters"], "eligible_count": len(eligible),
                      "selected_discovery": report["selected_discovery"]}, indent=2))

if __name__ == "__main__":
    main()
