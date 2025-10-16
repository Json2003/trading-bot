#!/usr/bin/env python3
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


if __name__ == "__main__":
    main()
