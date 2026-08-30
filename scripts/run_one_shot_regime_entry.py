#!/usr/bin/env python3
"""Evaluate one frozen one-shot regime-entry hypothesis; research-only."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from scripts.run_cross_asset_regime import (
    BLOCKS,
    CRYPTO_ASSETS,
    DISCOVERY_END,
    END,
    MACRO_ASSETS,
    START,
    _summary,
    _trade,
    _gate,
    ema_series,
    load_close_series,
    load_crypto_bars,
    regime_at,
)

def evaluate_one_shot(macro, emas, bars, start: date, end: date, asset: str):
    signal_dates = sorted(set.intersection(*(set(macro[name]) for name in MACRO_ASSETS)))
    signal_dates = [current for current in signal_dates if start <= current < end]
    rows: list[dict[str, Any]] = []
    previous_state = 0
    for current in signal_dates:
        state = regime_at(macro, emas, current)
        if state is None:
            continue
        direction = state[0]
        is_transition = direction != 0 and direction != previous_state
        if is_transition:
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
    emas = {name: ema_series(macro[name], 50) for name in MACRO_ASSETS}
    bars = {name: load_crypto_bars(args.data_dir / f"{name}.csv") for name in CRYPTO_ASSETS}

    candidates = {}
    for asset in CRYPTO_ASSETS:
        discovery_rows, discovery = evaluate_one_shot(macro, emas, bars, START, DISCOVERY_END, asset)
        holdout_rows, holdout = evaluate_one_shot(macro, emas, bars, DISCOVERY_END, END, asset)
        candidates[asset] = {
            "discovery": discovery,
            "holdout": holdout,
            "passes_discovery": _gate(discovery),
            "passes_confirmation": bool(_gate(discovery) and _gate(holdout)),
            "status": "confirmed" if _gate(discovery) and _gate(holdout) else "not_confirmed",
            "discovery_trade_rows": discovery_rows,
            "holdout_trade_rows": holdout_rows,
        }

    manifests = {
        name: json.loads((args.data_dir / f"{name}.manifest.json").read_text(encoding="utf-8"))
        for name in (*MACRO_ASSETS, *CRYPTO_ASSETS)
    }
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
            "inherited_from": "scripts/run_cross_asset_regime.py",
            "entry": "first BTC/ETH daily bar after the first qualifying signal following a completed daily regime transition plus one latency bar",
            "one_trade_per_regime_transition": True,
            "transition_reset": "any neutral day resets the regime state",
            "thresholds_unchanged": True,
            "horizon_unchanged": True,
            "notional": 3000.0,
        },
        "window": {
            "start": START.isoformat(),
            "discovery_end_exclusive": DISCOVERY_END.isoformat(),
            "end_exclusive": END.isoformat(),
            "holdout_untouched": True,
            "completed_bars_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
        },
        "source": {"manifests": manifests, "missing_data_is_excluded": True},
        "candidates": candidates,
        "status": "confirmed" if any(v["passes_confirmation"] for v in candidates.values()) else "not_confirmed",
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
        } for asset, value in candidates.items()
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
