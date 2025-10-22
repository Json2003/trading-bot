#!/usr/bin/env python3
<<<<<<< HEAD
"""
Train a LightGBM classifier on OHLCV features and save the model/report.

Usage examples:
  python scripts/train_lightgbm.py --source csv --path tradingbot_ibkr/datafiles/BTC_USDT_bars.csv
  python scripts/train_lightgbm.py --source ccxt --exchange binance --symbol "BTC/USDT" --timeframe 1h --since 2024-01-01

Notes:
  - Uses feature extraction from tradingbot_ibkr.feature_extraction (MA/RSI).
  - Avoids local shadowing of third-party deps (e.g., pandas.py) via import_third_party.
  - Saves model and a small report to tradingbot_ibkr/model_store/.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

# Keep repo root on sys.path for local package imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


def import_third_party(mod_name: str):
    """Import a third-party module ensuring repo-local files don't shadow it.

    Mirrors scripts/run_backtest.py strategy.
    """
    import importlib
    original = sys.path.copy()
    try:
        repo_paths = {p for p in original if REPO_ROOT in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        return importlib.import_module(mod_name)
    finally:
        sys.path = original


def parse_date(d: str) -> datetime:
    d = d.strip()
    if d.isdigit():
        ts = int(d)
        if ts > 10_000_000_000:  # ms
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date: {d}") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LightGBM on OHLCV features")
    p.add_argument("--source", choices=["csv", "ccxt"], required=True, help="Data source")
    # CSV
    p.add_argument("--path", help="CSV path with ts/open/high/low/close[/volume]")
    # CCXT
    p.add_argument("--exchange", default="binance", help="CCXT exchange id")
    p.add_argument("--symbol", default="BTC/USDT", help="Symbol, e.g., BTC/USDT")
    p.add_argument("--timeframe", default="1h", help="Timeframe for CCXT fetch")
    p.add_argument("--since", type=parse_date, help="Start date (YYYY-MM-DD or epoch)")
    p.add_argument("--until", type=parse_date, help="End date (YYYY-MM-DD or epoch)")

    # Labeling / model
    p.add_argument("--horizon", type=int, default=1, help="Future bars to look ahead for label")
    p.add_argument("--threshold", type=float, default=0.0, help="Return threshold for class=1")
    p.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds")
    p.add_argument("--learning-rate", type=float, default=0.05, help="LGBM learning rate")
    p.add_argument("--num-leaves", type=int, default=31, help="LGBM num_leaves")
    p.add_argument("--n-estimators", type=int, default=300, help="LGBM estimators")

    # Output
    p.add_argument("--model-out", default=None, help="Path to save model (default: model_store/lgbm_classifier.pkl)")
    p.add_argument("--report-out", default=None, help="Path to save JSON report (default: model_store/lgbm_report.json)")
    return p


def load_csv(path: str):
    pd = import_third_party("pandas")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "ts" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        except Exception:
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
        df = df.set_index("ts")
    elif df.index.name:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise ValueError("CSV must include 'ts' column or datetime index")

    lower_cols = {c.lower(): c for c in df.columns}
    req = ["open", "high", "low", "close"]
    if any(c not in lower_cols for c in req):
        # try case-insensitive rename
        df = df.rename(columns={c: c.lower() for c in df.columns})
        for c in req:
            if c not in df.columns:
                raise ValueError(f"CSV missing column: {c}")
    else:
        df = df.rename(columns=lower_cols)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


def fetch_ccxt(exchange: str, symbol: str, timeframe: str, since: Optional[datetime], until: Optional[datetime]):
    ccxt = import_third_party("ccxt")
    pd = import_third_party("pandas")
    ex = getattr(ccxt, exchange)()
    limit = 1000
    rows = []
    since_ms = int(since.timestamp() * 1000) if since else None
    until_ms = int(until.timestamp() * 1000) if until else None
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
        if until_ms and since_ms >= until_ms:
            break
    if not rows:
        raise RuntimeError("No OHLCV fetched")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def make_features(df):
    # Local modules (absolute import)
    from tradingbot_ibkr.feature_extraction import technical_indicators
    feat = technical_indicators(df)
    # Keep known feature columns
    cols = [c for c in ["ma_fast", "ma_slow", "rsi", "volume"] if c in feat.columns]
    return feat[cols], feat


def label_from_returns(df, horizon: int, threshold: float):
    import numpy as np
    close = df["close"]
    future = close.shift(-horizon)
    ret = (future - close) / close
    y = (ret > threshold).astype(int)
    return y


@dataclass
class TrainReport:
    model_name: str
    n_samples: int
    start: str
    end: str
    features: List[str]
    params: dict
    cv_mean_acc: float
    cv_std_acc: float


def train_lgbm(X, y, cv_splits: int, learning_rate: float, num_leaves: int, n_estimators: int):
    lgb = import_third_party("lightgbm")
    skmodel = lgb.LGBMClassifier(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )
    # Cross-validate with time series split
    sk = import_third_party("sklearn")
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    tscv = TimeSeriesSplit(n_splits=max(2, cv_splits))
    scores = cross_val_score(skmodel, X, y, cv=tscv, scoring="accuracy")
    # Fit final on full data
    skmodel.fit(X, y)
    return skmodel, float(scores.mean()), float(scores.std())


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    pd = import_third_party("pandas")
    joblib = import_third_party("joblib")

    # Load data
    if args.source == "csv":
        if not args.path:
            raise SystemExit("--path is required for --source csv")
        df = load_csv(args.path)
    else:
        df = fetch_ccxt(args.exchange, args.symbol, args.timeframe, args.since, args.until)

    # Build features and labels
    X, feat_full = make_features(df)
    y = label_from_returns(feat_full.join(df[["close"]], how="left"), args.horizon, args.threshold)
    # Align shapes
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if len(X) < 100:
        raise SystemExit("Not enough samples after feature/label alignment (need >= 100)")

    model, cv_mean, cv_std = train_lgbm(
        X,
        y,
        cv_splits=args.cv_splits,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
    )

    # Prepare outputs
    model_store = Path(REPO_ROOT) / "tradingbot_ibkr" / "model_store"
    model_store.mkdir(parents=True, exist_ok=True)
    model_path = args.model_out or str(model_store / "lgbm_classifier.pkl")
    report_path = args.report_out or str(model_store / "lgbm_report.json")

    joblib.dump(model, model_path)

    report = TrainReport(
        model_name="LightGBMClassifier",
        n_samples=int(len(X)),
        start=X.index.min().isoformat() if hasattr(X.index, 'isoformat') else str(X.index.min()),
        end=X.index.max().isoformat() if hasattr(X.index, 'isoformat') else str(X.index.max()),
        features=list(X.columns),
        params={
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "n_estimators": args.n_estimators,
            "horizon": args.horizon,
            "threshold": args.threshold,
            "cv_splits": args.cv_splits,
        },
        cv_mean_acc=cv_mean,
        cv_std_acc=cv_std,
    )
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print("Saved model:", model_path)
    print("Saved report:", report_path)
    print(json.dumps(asdict(report), indent=2))
=======
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

    df = pl.concat(frames).sort(["ts", "symbol"]).collect()
    if df.height == 0:
        raise DataLoadError(
            "Loaded dataframe is empty. Verify parquet inputs and symbol names."
        )
    return df


def to_numpy(df: pl.DataFrame, feature_columns: Sequence[str], label_column: str) -> tuple[np.ndarray, np.ndarray]:
    filtered = df.drop_nulls()
    X = filtered.select(feature_columns).to_numpy()
    y_series = filtered.get_column(label_column)
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

        best_iter = int(booster.best_iteration or num_boost_round)
        best_iterations.append(best_iter)
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
>>>>>>> origin/main


if __name__ == "__main__":
    main()
