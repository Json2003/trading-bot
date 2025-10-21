#!/usr/bin/env python3
"""CLI for running Optuna-based parameter searches on the SMA strategy."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import optuna

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Ensure real third-party packages take precedence over repo-local stubs
sys.path = [p for p in sys.path if p not in ("", REPO_ROOT)] + [REPO_ROOT]

for _name in ("pandas", "requests", "ccxt"):
    _mod = sys.modules.get(_name)
    if _mod is not None:
        _file = getattr(_mod, "__file__", "") or ""
        try:
            if REPO_ROOT in os.path.abspath(_file):
                del sys.modules[_name]
        except Exception:
            pass

from backtest.engine import ExecConfig, recommended_exec_config
from backtest.io import load_csv
from backtest.optimization import FEATURE_CHOICES, StrategyParams, make_objective, run_trial


def build_exec_config(args: argparse.Namespace) -> ExecConfig:
    cfg = recommended_exec_config()
    cfg.fees_bps = float(args.fees_bps)
    cfg.slip_bps = float(args.slip_bps)
    cfg.risk_per_trade = float(args.risk_per_trade)
    cfg.notional = float(args.notional)
    return cfg


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog='Available feature mixes: ' + ', '.join(FEATURE_CHOICES))
    parser.add_argument(
        "--data",
        default=os.path.join("backtest", "sample_data", "sample_ohlcv.csv"),
        help="Path to an OHLCV CSV file",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials to run")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional wall-clock timeout in seconds",
    )
    parser.add_argument(
        "--metric",
        choices=["sharpe", "sortino", "profit_factor", "total_return"],
        default="sharpe",
        help="Performance metric to maximise",
    )
    parser.add_argument("--storage", help="Optuna storage URI, e.g. sqlite:///study.db")
    parser.add_argument("--study-name", help="Study name when using persistent storage")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study if present")
    parser.add_argument("--seed", type=int, help="Seed for the TPE sampler")
    parser.add_argument("--fees-bps", type=float, default=5.0, help="Fees per fill in basis points")
    parser.add_argument("--slip-bps", type=float, default=2.0, help="Slippage per fill in basis points")
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
        help="Risk per trade as a fraction of equity",
    )
    parser.add_argument("--notional", type=float, default=1.0, help="Starting notional/equity")
    return parser.parse_args(argv)


def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return load_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = load_dataset(args.data)

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.seed is not None else None
    study_kwargs = {"direction": "maximize"}
    if args.resume and not args.study_name:
        raise SystemExit('--resume requires --study-name when using persistent storage')

    if sampler is not None:
        study_kwargs["sampler"] = sampler
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["study_name"] = args.study_name
        study_kwargs["load_if_exists"] = bool(args.resume)

    study = optuna.create_study(**study_kwargs)

    cfg = build_exec_config(args)
    objective = make_objective(df, cfg, metric=args.metric)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        catch=(ValueError,),
    )

    try:
        best = study.best_trial
    except ValueError:
        print("No successful trials were completed.")
        return 1

    params = StrategyParams(
        window=int(best.params["window"]),
        feature_mix=str(best.params["feature_mix"]),
        threshold=float(best.params["thr"]),
    )
    score, summary = run_trial(df, params, cfg, metric=args.metric, return_summary=True)

    print(f"Best trial #{best.number}")
    print(
        f"Params: window={best.params['window']}, feature_mix={best.params['feature_mix']}, "
        f"thr={best.params['thr']:.3f}"
    )
    print(f"{args.metric.title()} = {score:.4f}")
    print("Additional metrics:")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
