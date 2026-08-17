#!/usr/bin/env python3
"""Run a three-year, paper-only v3 backtest with explicit execution budgets.

The strategy engine is unchanged and receives conservative effective fee and
slippage budgets. The report separately preserves spread, impact, latency,
partial-fill, funding, and outage assumptions so reviewers can distinguish
measured engine P&L from execution-model stress assumptions.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Mapping

try:
    from scripts.execution_model import BASE_EXECUTION, STRESS_EXECUTION, ExecutionModel
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_v3_exploration import candidate_definitions
    from scripts.run_momentum_volatility_v3 import align_pair, build_pair_features, run_pair
except ModuleNotFoundError:
    from execution_model import BASE_EXECUTION, STRESS_EXECUTION, ExecutionModel
    from run_momentum_volatility_research import load_bars
    from run_v3_exploration import candidate_definitions
    from run_momentum_volatility_v3 import align_pair, build_pair_features, run_pair

SATOSHIS_PER_BTC = 100_000_000
THREE_YEARS_HOURS = 3 * 365 * 24

def _safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value

def _run(pair, config, model: ExecutionModel, *, initial_balance: float, order_notional: float):
    features = build_pair_features(pair, config)
    result = run_pair(
        pair, initial_balance=initial_balance, order_notional=order_notional,
        fees_bps=model.effective_fees_bps_per_side,
        slippage_bps=model.effective_slippage_bps_per_side,
        config=config, feature_map=features,
    )
    turnover = float(result.get("trades", 0)) * order_notional
    estimated_cost = turnover * model.round_trip_bps / 10_000.0
    return {
        "engine_result": result,
        "execution_model": model.as_dict(),
        "estimated_turnover_quote": turnover,
        "estimated_execution_cost_quote": estimated_cost,
        "cost_note": "Estimated from completed entry/exit count; exact fills require fill-ledger integration.",
    }

def run_backtest(btc_path: Path, eth_path: Path, output: Path, *, initial_balance: float, order_notional: float):
    btc, eth = load_bars(btc_path), load_bars(eth_path)
    pair = align_pair(btc, eth)
    if len(pair) < THREE_YEARS_HOURS:
        raise ValueError(f"need at least {THREE_YEARS_HOURS} aligned hourly candles")
    pair = pair[-THREE_YEARS_HOURS:]
    reports = {}
    for name, config in candidate_definitions().items():
        reports[name] = {
            "base": _run(pair, config, BASE_EXECUTION, initial_balance=initial_balance, order_notional=order_notional),
            "stress": _run(pair, config, STRESS_EXECUTION, initial_balance=initial_balance, order_notional=order_notional),
        }
    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "automatic_promotion": False,
        "window": {
            "bars": len(pair), "hours": len(pair), "years": len(pair) / (365 * 24),
            "start": pair[0].timestamp.isoformat(), "end": pair[-1].timestamp.isoformat(),
            "completed_candles_only": True, "forward_test_reused_for_training": False,
        },
        "denomination": {
            "btc_satoshis_per_btc": SATOSHIS_PER_BTC,
            "btc_pnl_sats_available_when_starting_balance_is_btc": True,
            "quote_currency_results_preserved": True,
            "eth_sats_conversion": "requires timestamped BTC/ETH conversion; not inferred here",
        },
        "execution_models": {"base": BASE_EXECUTION.as_dict(), "stress": STRESS_EXECUTION.as_dict()},
        "candidates": reports,
        "status": "backtest_artifact_generated; exact fill ledger still required before claiming execution-realistic results",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_safe(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=75000.0)
    parser.add_argument("--order-notional", type=float, default=6000.0)
    args = parser.parse_args()
    report = run_backtest(args.btc_path, args.eth_path, args.output, initial_balance=args.initial_balance, order_notional=args.order_notional)
    print(json.dumps({"status": report["status"], "window": report["window"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
