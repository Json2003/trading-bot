#!/usr/bin/env python3
"""Train a LightGBM classifier on minute-level engineered features."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_FEATURE_DIR = Path("data/parquet/features_1m")
DEFAULT_LABEL_DIR = Path("data/parquet/labels_1m")
DEFAULT_SYMBOLS: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DEFAULT_FEATURE_COLUMNS: Sequence[str] = (
    "close",
    "ret_5m",
    "ret_30m",
    "vol_realized_30m",
    "zscore_30m",
    "rsi_14",
)
LABEL_COLUMN = "y"
JOIN_KEYS: Sequence[str] = ("symbol", "ts")


class DataLoadError(RuntimeError):
    """Raised when no parquet rows could be loaded for the requested symbols."""


def _parse_symbols(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        return [s for s in (part.strip() for part in raw.split(",")) if s]
    return list(raw)


def load_dataset(
    feature_dir: Path,
    label_dir: Path,
    symbols: Sequence[str],
    feature_columns: Sequence[str],
    label_column: str = LABEL_COLUMN,
) -> pl.DataFrame:
    """Load feature and label parquet files and return a joined Polars DataFrame."""

    frames: list[pl.LazyFrame] = []
    for symbol in symbols:
        feature_glob = feature_dir / f"symbol={symbol}" / "date=*" / "part.parquet"
        label_path = label_dir / f"symbol={symbol}" / "labels.parquet"
        feat_scan = pl.scan_parquet(str(feature_glob))
        label_scan = pl.scan_parquet(str(label_path))
        frame = (
            feat_scan.join(label_scan, on=list(JOIN_KEYS), how="inner")
            .select(list(JOIN_KEYS) + list(feature_columns) + [label_column])
        )
        frames.append(frame)

    if not frames:
        raise DataLoadError("No symbols provided for dataset loading.")

    df = pl.concat(frames).sort(list(JOIN_KEYS)).collect()
    if df.height == 0:
        raise DataLoadError(
            "Loaded dataframe is empty. Verify parquet inputs and symbol names."
        )
    return df


def to_numpy(df: pl.DataFrame, feature_columns: Sequence[str], label_column: str) -> tuple[np.ndarray, np.ndarray]:
    filtered = df.drop_nulls()
    X = filtered.select(feature_columns).to_numpy()
    y_series = filtered.get_column(label_column)
    # Expect binary labels encoded as 0 (negative class) and 1 (positive class).
    # Validate label encoding.
    unique_labels = set(y_series.unique().to_list())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column '{label_column}' contains non-binary values: {unique_labels}. "
            "Expected only 0 and 1."
        )
    y = (y_series == 1).to_numpy().astype(np.uint8)
    return X, y


def run_time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    num_boost_round: int,
    n_splits: int,
    threshold: float,
    early_stopping_rounds: int,
) -> tuple[list[dict[str, float]], list[int]]:
    if len(X) < n_splits + 1:
        raise RuntimeError(
            "Not enough samples for the requested TimeSeriesSplit configuration."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    best_iterations: list[int] = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X[valid_idx], label=y[valid_idx], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=[f"fold_{fold}"],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        best_iter = int((booster.best_iteration or num_boost_round) - 1)
        best_iterations.append(best_iter + 1)  # Store 1-indexed value for reporting
        preds = booster.predict(X[valid_idx], num_iteration=best_iter)
        ap = average_precision_score(y[valid_idx], preds)
        f1 = f1_score(y[valid_idx], (preds > threshold).astype(int))
        fold_metrics.append(
            {
                "fold": int(fold),
                "average_precision": float(ap),
                "f1": float(f1),
                "best_iteration": int(best_iter),
            }
        )

    return fold_metrics, best_iterations


def train_full_model(X: np.ndarray, y: np.ndarray, params: dict, num_boost_round: int) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y)
    booster = lgb.train(params, dataset, num_boost_round=num_boost_round, verbose_eval=False)
    return booster


def build_lgbm_params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "min_data_in_leaf": args.min_data_in_leaf,
        "seed": args.seed,
        "bagging_seed": args.seed,
        "feature_fraction_seed": args.seed,
        "verbose": -1,
    }


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR, help="Directory that stores feature parquet partitions.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_DIR, help="Directory containing label parquet files.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated list of symbols to train on.")
    parser.add_argument("--feature-cols", default=",".join(DEFAULT_FEATURE_COLUMNS), help="Comma-separated list of feature columns to use.")
    parser.add_argument("--label-col", default=LABEL_COLUMN, help="Name of the label column in the joined dataset.")
    parser.add_argument("--splits", type=int, default=5, help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--num-boost-round", type=int, default=200, help="Maximum boosting rounds for LightGBM.")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold used for F1 computation.")
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--tag", default=None, help="Optional model tag; defaults to a timestamp-based name.")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols specified for training.")
    feature_columns = _parse_symbols(args.feature_cols)
    if not feature_columns:
        raise SystemExit("No feature columns specified for training.")

    np.random.seed(args.seed)

    try:
        df = load_dataset(args.features, args.labels, symbols, feature_columns, args.label_col)
    except DataLoadError as exc:
        raise SystemExit(str(exc)) from exc
    X, y = to_numpy(df, feature_columns, args.label_col)
    if len(X) == 0:
        raise SystemExit(
            "No samples available after dropping null rows. Check data coverage or feature selection."
        )

    params = build_lgbm_params(args)
    fold_metrics, best_iters = run_time_series_cv(
        X,
        y,
        params,
        args.num_boost_round,
        args.splits,
        args.threshold,
        args.early_stopping_rounds,
    )

    best_iter = int(round(float(np.mean(best_iters)))) if best_iters else args.num_boost_round
    best_iter = max(1, min(best_iter, args.num_boost_round))
    final_model = train_full_model(X, y, params, best_iter)

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or sanitize_tag(
        f"lgbm_{'-'.join(symbols)}_{int(time.time())}"
    )
    model_path = args.outdir / f"{tag}.txt"
    metrics_path = args.outdir / f"{tag}_metrics.json"

    final_model.save_model(str(model_path))
    metrics = {
        "symbols": symbols,
        "feature_columns": feature_columns,
        "label_column": args.label_col,
        "fold_metrics": fold_metrics,
        "average_precision_mean": float(np.mean([m["average_precision"] for m in fold_metrics])),
        "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
        "best_iterations": best_iters,
        "best_iteration": best_iter,
        "params": params,
        "threshold": args.threshold,
        "num_rows": int(len(X)),
        "features_path": str(args.features),
        "labels_path": str(args.labels),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Saved LightGBM model to {model_path} | AP={metrics['average_precision_mean']:.4f} "
        f"F1={metrics['f1_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
