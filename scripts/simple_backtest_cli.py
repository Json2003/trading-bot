#!/usr/bin/env python3
"""Lightweight CLI for running a basic backtest over CSV OHLCV data."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.append(str(REPO_ROOT))


def parse_key_value_args(pairs: Iterable[str]) -> Dict[str, Any]:
    """Parse key=value strings into a dictionary with simple type coercion."""

    result: Dict[str, Any] = {}
    for item in pairs:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid argument '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        value = value.strip()
        if value.isdigit():
            parsed: Any = int(value)
        else:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
        result[key.strip()] = parsed
    return result


def resolve_strategy(spec: str):
    try:
        module_name, func_name = spec.split(":", 1)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Invalid strategy spec '{spec}'. Use module:function"
        ) from exc
    module = importlib.import_module(module_name)
    try:
        fn = getattr(module, func_name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(
            f"Strategy '{func_name}' not found in module '{module_name}'"
        ) from exc
    return fn


def run_backtest(args: argparse.Namespace) -> None:
    if args.source != "csv":
        raise ValueError("Only CSV source supported for now")
    if not args.path:
        raise ValueError("--path is required when --source csv")

    data_path = Path(args.path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    if df.empty:
        logger.error("Empty DataFrame loaded from %s", data_path)
        raise ValueError("DataFrame is empty")

    strat_defaults = {
        "fast": 8,
        "slow": 34,
        "trend_fast": 55,
        "trend_slow": 144,
    }
    strat_args = strat_defaults.copy()
    strat_args.update(parse_key_value_args(args.strategy_args))
    try:
        max_period = max(
            int(strat_args.get("fast", strat_defaults["fast"])),
            int(strat_args.get("slow", strat_defaults["slow"])),
            int(strat_args.get("trend_fast", strat_defaults["trend_fast"])),
            int(strat_args.get("trend_slow", strat_defaults["trend_slow"])),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Strategy period arguments must be numeric") from exc

    if args.max_bars and args.max_bars < max_period:
        logger.error(
            "max_bars (%d) too low for strategy periods (need >= %d)",
            args.max_bars,
            max_period,
        )
        raise ValueError(f"max_bars must be >= {max_period}")

    if args.max_bars:
        df = df.iloc[: args.max_bars].copy()

    strategy_fn = resolve_strategy(args.strategy)
    logger.info("Strategy args: %s", strat_args)
    df = strategy_fn(df, **strat_args)

    positions = []
    equity = [float(args.notional)]
    current_pos = 0.0
    entry_price = 0.0
    sl_fraction = args.sl_bps / 10_000 if args.sl_bps else 0.0
    tp_fraction = args.tp_bps / 10_000 if args.tp_bps else 0.0
    fee_fraction = args.fees_bps / 10_000 if args.fees_bps else 0.0
    slip_fraction = args.slip_bps / 10_000 if args.slip_bps else 0.0

    for timestamp, row in df.iterrows():
        signal = row.get(signal_col, 0)
        if pd.isna(signal):
            signal = 0
        price = float(row["close"])
        fees = fee_fraction * price
        slip = slip_fraction * price
        adjusted_price = price + slip if signal == 1 else price - slip

        if signal == 1 and current_pos == 0.0:
            if sl_fraction > 0:
                risk_amount = args.risk_per_trade * equity[-1]
                denominator = price * sl_fraction
                size = risk_amount / denominator if denominator else 0.0
            else:
                size = equity[-1] / price if price else 0.0
            current_pos = size
            entry_price = adjusted_price + fees
        elif signal == -1 and current_pos > 0.0:
            profit = current_pos * (adjusted_price - entry_price - fees)
            equity.append(equity[-1] + profit)
            positions.append(
                {
                    "entry": entry_price,
                    "exit": adjusted_price,
                    "profit": profit,
                    "exit_time": timestamp,
                    "reason": "signal",
                }
            )
            current_pos = 0.0

        if current_pos > 0.0:
            take_profit = entry_price * (1 + tp_fraction) if tp_fraction else None
            stop_loss = entry_price * (1 - sl_fraction) if sl_fraction else None

            if take_profit and price >= take_profit:
                profit = current_pos * (price - entry_price - fees)
                equity.append(equity[-1] + profit)
                positions.append(
                    {
                        "entry": entry_price,
                        "exit": price,
                        "profit": profit,
                        "exit_time": timestamp,
                        "reason": "tp",
                    }
                )
                current_pos = 0.0
            elif stop_loss and price <= stop_loss:
                profit = current_pos * (price - entry_price - fees)
                equity.append(equity[-1] + profit)
                positions.append(
                    {
                        "entry": entry_price,
                        "exit": price,
                        "profit": profit,
                        "exit_time": timestamp,
                        "reason": "sl",
                    }
                )
                current_pos = 0.0

    out_dir = Path(args.out_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    blotter = pd.DataFrame(positions)
    blotter.to_csv(out_dir / "blotter.csv", index=False)
    pd.Series(equity, name="equity").to_csv(out_dir / "equity_curve.csv", index=False)

    win_trades = int((blotter["profit"] > 0).sum()) if not blotter.empty else 0
    metrics = {
        "trades": len(blotter),
        "win_rate": win_trades / len(blotter) if len(blotter) else 0.0,
        "final_equity": equity[-1] if equity else 0.0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("Backtest complete. Outputs in %s", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--source", required=True, choices=["csv"], help="Data source (csv)")
    parser.add_argument("--path", help="Path to data")
    parser.add_argument(
        "--strategy",
        default="backtest.strategies.sma_filtered:generate_signals",
        help="Strategy module:function",
    )
    parser.add_argument(
        "--strategy_args",
        nargs="+",
        default=("fast=8", "slow=34", "trend_fast=55", "trend_slow=144"),
        help="Strategy args like fast=8",
    )
    parser.add_argument("--fees_bps", type=float, default=0.0, help="Fees in bps")
    parser.add_argument("--slip_bps", type=float, default=0.0, help="Slippage in bps")
    parser.add_argument("--tp_bps", type=float, default=0.0, help="Take profit in bps")
    parser.add_argument("--sl_bps", type=float, default=0.0, help="Stop loss in bps")
    parser.add_argument("--max_bars", type=int, default=0, help="Max bars to process")
    parser.add_argument("--notional", type=float, default=1.0, help="Initial notional")
    parser.add_argument("--risk_per_trade", type=float, default=0.01, help="Risk per trade")
    parser.add_argument("--out_prefix", required=True, help="Output prefix")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":  # pragma: no cover
    main()
