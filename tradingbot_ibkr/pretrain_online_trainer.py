"""Pretrain the OnlineTrainer on historical bar CSVs.

This script loads CSV files (default: datafiles/*.csv), constructs simple features
and a next-bar up/down label, then incrementally trains the `OnlineTrainer`.

Usage (PowerShell):
python pretrain_online_trainer.py --files datafiles/*.csv --epochs 2 --max-samples 100000
"""
import argparse
from pathlib import Path
import pandas as pd
from models.online_trainer import OnlineTrainer
from river import ensemble, preprocessing, tree
import json
import time
from tradingbot_ibkr.data import store
from tradingbot_ibkr.data import store


def build_examples_from_df(df, max_samples=None, feature_cols=None):
    # require at least 2 rows to create label
    if len(df) < 2:
        return []
    df = df.copy()
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    # simple features: close, high, low, volume, pct change, 3-bar MA
    # basic returns and moving averages
    df['ret1'] = df['close'].pct_change().fillna(0.0)
    df['ma3'] = df['close'].rolling(3).mean().fillna(method='bfill')
    # momentum features
    df['mom5'] = df['close'].pct_change(5).fillna(0.0)
    df['mom10'] = df['close'].pct_change(10).fillna(0.0)
    # volume-based features
    df['vol_mean20'] = df['volume'].rolling(20).mean().fillna(method='bfill') if 'volume' in df.columns else 0.0
    df['vol_ratio'] = df['volume'] / (df['vol_mean20'].replace(0, 1))
    # volatility
    df['vol20'] = df['ret1'].rolling(20).std().fillna(0.0)
    # ATR (14)
    high_low = (df['high'] - df['low']).abs()
    high_pc = (df['high'] - df['close'].shift(1)).abs()
    low_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean().fillna(method='bfill')
    # RSI (14)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean().replace(0, 1e-8)
    rs = roll_up / roll_down
    df['rsi14'] = 100.0 - (100.0 / (1.0 + rs))
    examples = []
    for idx, row in df.iterrows():
        # build features according to requested columns
        if feature_cols:
            feat = {col: float(row.get(col, 0.0)) for col in feature_cols}
        else:
            feat = {
                'close': float(row['close']),
                'high': float(row['high']),
                'low': float(row['low']),
                'volume': float(row.get('volume', 0.0)),
                'ret1': float(row['ret1']),
                'ma3': float(row['ma3']),
                'mom5': float(row['mom5']),
                'mom10': float(row['mom10']),
                'vol20': float(row['vol20']),
                'vol_ratio': float(row['vol_ratio']),
                'atr14': float(row['atr14']),
                'rsi14': float(row['rsi14'])
            }
        # label will be computed externally based on horizon in main()
        examples.append((feat, None))
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def label_examples_with_horizon(df, examples, horizon=12, profit_pct=0.01):
    # compute future close at horizon
    labels = []
    closes = df['close'].values
    for i in range(len(examples)):
        future_idx = i + horizon
        if future_idx < len(closes):
            labels.append(1 if closes[future_idx] > closes[i] * (1 + profit_pct) else 0)
        else:
            labels.append(0)
    return labels
    


def run_pretrain(paths, epochs=1, max_samples=None, threshold=0.5, feature_cols=None, horizon=12, profit_pct=0.01, job_file: str = None):
    """Programmatic pretrain entry. Writes progress/status to job_file (if provided).

    paths: list of CSV file paths
    Returns: dict with summary
    """
    trainer = OnlineTrainer()
    trainer.model = preprocessing.StandardScaler() | ensemble.BaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42)
    try:
        trainer.load()
    except Exception:
        pass

    total_examples = 0
    correct = 0
    seen = 0
    dist = {'pred1': 0, 'pred0': 0}

    # feature_cols may be precomputed list
    for epoch in range(epochs):
        for i, path in enumerate(paths):
            df = pd.read_csv(path, parse_dates=['ts'])
            if 'ts' in df.columns:
                df.set_index('ts', inplace=True)
            examples = build_examples_from_df(df, max_samples=max_samples, feature_cols=feature_cols)
            labels = label_examples_with_horizon(df, examples, horizon=horizon, profit_pct=profit_pct)
            n = len(examples)
            # report starting file progress
            if job_file:
                _write_job_update(job_file, status='running', progress=0, message=f"epoch {epoch+1} file {i+1}/{len(paths)}: {n} examples")
            for idx, ((feat, _), label) in enumerate(zip(examples, labels)):
                prob = trainer.predict_proba(feat)
                pred = 1 if prob >= threshold else 0
                seen += 1
                if pred == 1:
                    dist['pred1'] += 1
                else:
                    dist['pred0'] += 1
                if pred == label:
                    correct += 1
                trainer.learn_one(feat, label)
                total_examples += 1
                # periodic progress write
                if job_file and total_examples % 500 == 0:
                    pct = None
                    try:
                        pct = (total_examples / max(1, len(examples) * epochs * len(paths))) * 100
                    except Exception:
                        pct = 0
                    _write_job_update(job_file, status='running', progress=pct, message=f'seen {total_examples} examples')
    # final save
    trainer.save()
    try:
        import pickle
        model_artifact = Path(__file__).resolve().parents[0] / 'model_store' / 'online_model.pkl'
        model_artifact.parent.mkdir(parents=True, exist_ok=True)
        with open(model_artifact, 'wb') as f:
            pickle.dump(trainer.model, f)
        mv = store.get_or_update_model_version()
    except Exception as e:
        mv = None
        if job_file:
            _write_job_update(job_file, status='error', progress=100, message=f'model save failed: {e}')

    acc = (correct / seen) if seen else None
    summary = {
        'total_examples': total_examples,
        'prediction_distribution': dist,
        'accuracy': acc,
        'model_version': mv
    }
    if job_file:
        _write_job_update(job_file, status='done', progress=100, result=summary)
    return summary


def _write_job_update(job_file, **data):
    try:
        p = Path(job_file)
        j = {}
        if p.exists():
            try:
                j = json.loads(p.read_text())
            except Exception:
                j = {}
        j.update(data)
        p.write_text(json.dumps(j, indent=2))
    except Exception:
        pass


def main():
    p = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', default=[str(p / 'datafiles' / '*.csv')], help='CSV files or glob(s) to load')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.5, help='reporting threshold for online predictions')
    parser.add_argument('--features', type=str, default=None, help='comma-separated feature columns to use (default basic set)')
    parser.add_argument('--horizon', type=int, default=12, help='label horizon in bars to look ahead')
    parser.add_argument('--profit-pct', type=float, default=0.01, help='profit pct used to define positive label')
    parser.add_argument('--job-file', help='optional path to job file to write progress')
    args = parser.parse_args()

    # expand globs
    from glob import glob
    paths = []
    for pattern in args.files:
        paths.extend(glob(pattern))
    if not paths:
        print('No files found for', args.files)
        return

    feature_cols = [c.strip() for c in args.features.split(',')] if args.features else None
    summary = run_pretrain(paths, epochs=args.epochs, max_samples=args.max_samples, threshold=args.threshold, feature_cols=feature_cols, horizon=args.horizon, profit_pct=args.profit_pct, job_file=args.job_file)
    print('Pretrain summary:', summary)


if __name__ == '__main__':
    main()
