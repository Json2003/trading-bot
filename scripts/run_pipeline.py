#!/usr/bin/env python3
"""
End-to-end automation runner for the Splitstar Operations Console.

Steps:
 1. Optional feature ingestion (macro + news) into the feature store.
 2. Optional LightGBM training on CCXT OHLCV data with report/model outputs.
 3. Optional aggressive strategy backtest with JSON report.

Usage example:
    python scripts/run_pipeline.py --symbol BTC/USDT --timeframe 1h --since 2024-01-01

Outputs are collected under automation_runs/<timestamp>/ by default.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repo root importable for local packages
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate ingestion, training, and backtesting workflow.")
    parser.add_argument("--exchange", default="binance", help="CCXT exchange id (default: binance)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbol to train/backtest (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe for OHLCV fetch (default: 1h)")
    parser.add_argument("--since", default="2024-01-01", help="Start date (YYYY-MM-DD or epoch) for training/backtest")
    parser.add_argument("--until", default=None, help="Optional end date (YYYY-MM-DD or epoch)")
    parser.add_argument("--start-balance", type=float, default=10_000.0, help="Starting balance for backtest")
    parser.add_argument("--tp", type=float, default=0.004, help="Take profit fraction for backtest")
    parser.add_argument("--sl", type=float, default=0.002, help="Stop loss fraction for backtest")
    parser.add_argument("--hold", type=int, default=12, help="Max holding bars for backtest")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip feature ingestion step")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest step")
    parser.add_argument("--output-dir", default="automation_runs", help="Base directory for outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def ensure_output_dir(base: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_ingestion(run_dir: Path) -> Dict[str, Any]:
    from feature_registry import FeatureRegistryService, MacroEconomicPipeline, NewsEmbeddingsPipeline
    from feature_registry.storage import FeatureStore

    store = FeatureStore(root=run_dir / "feature_store")
    service = FeatureRegistryService(
        pipelines=[
            MacroEconomicPipeline(start_year=2018),
            NewsEmbeddingsPipeline(max_items=20),
        ],
        store=store,
    )
    results = service.run_all()
    ingest_report = [
        {
            "name": r.name,
            "as_of": r.as_of.isoformat(),
            "data_version": r.data_version,
            "row_count": r.row_count,
            "artifact_path": r.artifact_path,
            "metadata": r.metadata,
        }
        for r in results
    ]
    (run_dir / "ingestion_results.json").write_text(json.dumps(ingest_report, indent=2))
    return {"pipelines": ingest_report}


def run_training(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    from scripts import train_lightgbm as train_lgb

    since = args.since
    until = args.until

    df = train_lgb.fetch_ccxt(args.exchange, args.symbol, args.timeframe, train_lgb.parse_date(since) if since else None, train_lgb.parse_date(until) if until else None)
    X, feat_full = train_lgb.make_features(df)
    enriched = feat_full.join(df[["close"]], how="left")
    y = train_lgb.label_from_returns(enriched, horizon=1, threshold=0.0)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    if len(X) < 100:
        raise RuntimeError(f"Not enough samples after feature/label alignment (got {len(X)}, need >= 100)")

    model, cv_mean, cv_std = train_lgb.train_lgbm(
        X,
        y,
        cv_splits=5,
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=300,
    )

    joblib = train_lgb.import_third_party("joblib")
    model_path = run_dir / "lgbm_classifier.pkl"
    report_path = run_dir / "lgbm_report.json"
    joblib.dump(model, model_path)

    report = train_lgb.TrainReport(
        model_name="LightGBMClassifier",
        n_samples=int(len(X)),
        start=X.index.min().isoformat() if hasattr(X.index, "isoformat") else str(X.index.min()),
        end=X.index.max().isoformat() if hasattr(X.index, "isoformat") else str(X.index.max()),
        features=list(X.columns),
        params={
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 300,
            "horizon": 1,
            "threshold": 0.0,
            "cv_splits": 5,
        },
        cv_mean_acc=cv_mean,
        cv_std_acc=cv_std,
    )
    report_dict = asdict(report)
    report_path.write_text(json.dumps(report_dict, indent=2))
    return {
        "model_path": str(model_path),
        "report_path": str(report_path),
        "cv_mean_acc": cv_mean,
        "cv_std_acc": cv_std,
        "n_samples": report.n_samples,
        "feature_count": len(report.features),
    }


def run_backtest(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    from scripts import run_backtest as backtest_cli
    bt_args = backtest_cli.build_parser().parse_args(
        [
            "--source",
            "ccxt",
            "--exchange",
            args.exchange,
            "--symbol",
            args.symbol,
            "--timeframe",
            args.timeframe,
            "--since",
            args.since,
            "--tp",
            str(args.tp),
            "--sl",
            str(args.sl),
            "--hold",
            str(args.hold),
            "--start-balance",
            str(args.start_balance),
            "--out",
            str(run_dir / "backtest_report.json"),
        ]
        + (["--until", args.until] if args.until else [])
    )
    stats = backtest_cli.run_backtest(bt_args)
    (run_dir / "backtest_summary.json").write_text(json.dumps(stats, indent=2, default=str))
    return {
        "trades": stats.get("trades"),
        "pnl": stats.get("pnl"),
        "total_return_pct": stats.get("total_return_pct"),
        "annual_roi_pct": stats.get("annual_roi_pct"),
        "win_rate_pct": stats.get("win_rate_pct"),
        "report_path": str(run_dir / "backtest_report.json"),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_dir = ensure_output_dir(args.output_dir)
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "exchange": args.exchange,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    try:
        if not args.skip_ingest:
            logging.info("Running feature ingestion...")
            summary["steps"]["ingestion"] = run_ingestion(run_dir)
            logging.info("Feature ingestion completed.")
        else:
            logging.info("Skipping ingestion step.")
    except Exception as exc:
        logging.exception("Feature ingestion failed")
        summary["steps"]["ingestion"] = {"error": str(exc)}

    try:
        if not args.skip_train:
            logging.info("Running model training...")
            summary["steps"]["training"] = run_training(args, run_dir)
            logging.info("Model training completed.")
        else:
            logging.info("Skipping training step.")
    except Exception as exc:
        logging.exception("Model training failed")
        summary["steps"]["training"] = {"error": str(exc)}

    try:
        if not args.skip_backtest:
            logging.info("Running backtest...")
            summary["steps"]["backtest"] = run_backtest(args, run_dir)
            logging.info("Backtest completed.")
        else:
            logging.info("Skipping backtest step.")
    except Exception as exc:
        logging.exception("Backtest failed")
        summary["steps"]["backtest"] = {"error": str(exc)}

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Summary written to %s", summary_path)

    print("\nAutomation Summary")
    print("------------------")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
