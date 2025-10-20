#!/usr/bin/env python3
"""Convenience helper to sweep pretraining hyper-parameters and validate winners.

Example:
  python scripts/pretrain_sweep.py \
      --files tradingbot_ibkr/datafiles/BTC_USDT_bars.csv \
              tradingbot_ibkr/datafiles/BTCUSDT_bars_sample.csv \
              tradingbot_ibkr/datafiles/BTC_USDT_bars_annotated.csv \
      --macro-files tradingbot_ibkr/datafiles/econ/CPIAUCSL_fred.csv \
                     tradingbot_ibkr/datafiles/econ/UNRATE_fred.csv \
      --horizons 3 4 \
      --profit-pcts 0.003 0.0035 \
      --roc-min 0.9 --win-min 0.8 --run-stress
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

if __package__ is None or __package__ == "":  # executed as script
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(Path.home() / '.local' / 'lib' / f"python{sys.version_info.major}.{sys.version_info.minor}" / 'site-packages'))

try:
    import pip._vendor.requests as _real_requests  # type: ignore
    sys.modules['requests'] = _real_requests
except Exception:
    pass

import json
import time

from tradingbot_ibkr.pretrain_online_trainer import run_pretrain
import tradingbot_ibkr.validate_candidates as validate_candidates


def run_validation(run_stress: bool) -> None:
    """Execute the standard validation (and optional stress test)."""
    print("\nRunning validate_candidates.py …", flush=True)
    validate_candidates.main()
    if run_stress:
        import tradingbot_ibkr.validate_candidates_stress as stress
        print("Running validate_candidates_stress.py …", flush=True)
        stress.main()


def sweep(args: argparse.Namespace) -> None:
    combos = list(itertools.product(args.horizons, args.profit_pcts))
    results = []
    for horizon, tp in combos:
        start = time.time()
        summary = run_pretrain(
            paths=args.files,
            epochs=args.epochs,
            max_samples=args.max_samples,
            threshold=args.train_threshold,
            horizon=horizon,
            profit_pct=tp,
            sl_pct=tp * args.sl_multiplier,
            balance=not args.no_balance,
            augment_factor=args.augment_factor,
            augment_noise=args.augment_noise,
            macro_files=args.macro_files,
            model_type=args.model_type,
            eval_threshold=args.eval_threshold,
            job_file=str(args.job_dir / f"auto_pretrain_h{horizon}_tp{tp:.4f}.json")
        )
        elapsed = time.time() - start
        results.append((horizon, tp, summary))
        win_rt = summary.get('win_rate_at_threshold')
        win_display = f"{win_rt:.2%}" if win_rt is not None else 'n/a'
        print(f"h={horizon} tp={tp:.4f} -> ROC={summary['roc_auc']:.3f} win@{args.eval_threshold:.2f}={win_display} hits={summary['threshold_hits']} total={summary['total_examples']} ({elapsed:.1f}s)")

        if (summary['roc_auc'] and summary['roc_auc'] >= args.roc_min and
                summary['win_rate_at_threshold'] and summary['win_rate_at_threshold'] >= args.win_min and
                summary['threshold_hits'] >= args.min_hits):
            run_validation(args.run_stress)

    print("\nSummary (sorted by ROC):")
    for horizon, tp, summary in sorted(results, key=lambda r: r[2]['roc_auc'] or 0, reverse=True):
        print(json.dumps({
            'horizon': horizon,
            'tp': tp,
            'roc_auc': summary['roc_auc'],
            'accuracy': summary['accuracy'],
            'win_rate_at_threshold': summary['win_rate_at_threshold'],
            'threshold_hits': summary['threshold_hits'],
            'total_examples': summary['total_examples']
        }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep pretraining hyper-parameters and validate winners.")
    parser.add_argument('--files', nargs='+', required=True, help='Training CSV files with OHLCV data')
    parser.add_argument('--macro-files', nargs='*', default=[], help='Optional macro CSV files (date,value)')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 4, 5], help='Label horizons to sweep')
    parser.add_argument('--profit-pcts', nargs='+', type=float, default=[0.003, 0.0035], help='Target TP percentages to sweep')
    parser.add_argument('--sl-multiplier', type=float, default=0.5, help='SL percentage expressed as multiplier of TP (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-samples', type=int, default=4000)
    parser.add_argument('--augment-factor', type=int, default=3)
    parser.add_argument('--augment-noise', type=float, default=0.02)
    parser.add_argument('--train-threshold', type=float, default=0.5, help='Probability threshold used during training stats')
    parser.add_argument('--eval-threshold', type=float, default=0.7, help='Probability threshold to gauge win rate')
    parser.add_argument('--roc-min', type=float, default=0.9)
    parser.add_argument('--win-min', type=float, default=0.8)
    parser.add_argument('--min-hits', type=int, default=30, help='Minimum number of eval-threshold samples before validation triggers')
    parser.add_argument('--no-balance', action='store_true', help='Disable dataset balancing')
    parser.add_argument('--model-type', choices=['logistic', 'forest'], default='forest')
    parser.add_argument('--run-stress', action='store_true', help='Run validate_candidates_stress after each winning pretrain')
    parser.add_argument('--output-dir', default='tradingbot_ibkr/model_store/jobs', help='Directory to write job JSON summaries')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.job_dir = Path(args.output_dir)
    args.job_dir.mkdir(parents=True, exist_ok=True)
    sweep(args)


if __name__ == '__main__':
    main()
