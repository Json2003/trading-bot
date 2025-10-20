#!/usr/bin/env python3
"""Nightly research harness for Optuna-driven multi-strategy tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.optimization.research_loop import (
    NightlyResearchLoop,
    create_non_overlapping_windows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=Path("backtest/sample_data/sample_ohlcv.csv"),
        help="Path to the OHLCV CSV used for optimisation windows",
    )
    parser.add_argument(
        "--registry",
        default=Path("configs/research_registry.json"),
        help="Destination JSON file storing the top-N configurations",
    )
    parser.add_argument("--window-size", type=int, default=600, help="Rows per optimisation window")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Fraction of each window reserved for out-of-sample scoring",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="Maximum number of non-overlapping windows to evaluate each run",
    )
    parser.add_argument(
        "--trials-min",
        type=int,
        default=20,
        help="Lower bound on Optuna trials per window",
    )
    parser.add_argument(
        "--trials-max",
        type=int,
        default=50,
        help="Upper bound on Optuna trials per window",
    )
    parser.add_argument(
        "--paper-threshold",
        type=float,
        default=0.05,
        help="Relative improvement required to flag configs for paper trading",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of best configurations retained in the registry",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatability")
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"close", "high", "low"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} missing required columns: {sorted(missing)}")
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    # Provide synthetic futures data if absent so the arbitrage simulator can run.
    if "spot_close" not in df.columns:
        df["spot_close"] = df["close"]
    if "futures_close" not in df.columns:
        df["futures_close"] = df["close"] * 1.001
    return df


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    df = load_dataset(data_path)
    windows = create_non_overlapping_windows(
        df,
        window_size=int(args.window_size),
        test_fraction=float(args.test_fraction),
        max_windows=int(args.windows),
    )
    if not windows:
        raise SystemExit("No valid optimisation windows produced")

    loop = NightlyResearchLoop(
        windows,
        Path(args.registry),
        trials_range=(int(args.trials_min), int(args.trials_max)),
        paper_improve_threshold=float(args.paper_threshold),
        top_n=int(args.top_n),
        seed=args.seed,
    )

    results = loop.run(max_windows=int(args.windows))
    if not results:
        print("No successful trials completed")
        return 1

    for entry in results:
        flag = " [PAPER]" if entry.flag_for_paper_trial else ""
        print(
            f"{entry.window}: score={entry.score:.3f} sharpe={entry.oos_sharpe:.3f} "
            f"dd={entry.oos_max_drawdown:.3f} turnover={entry.turnover:.3f}{flag}"
        )
    print(f"Registry updated at {Path(args.registry).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

