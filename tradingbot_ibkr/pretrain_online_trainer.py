"""Pretrain the OnlineTrainer on historical bar CSVs.

This script loads CSV files (default: datafiles/*.csv), constructs simple features
and a next-bar up/down label, then incrementally trains the `OnlineTrainer`.

Usage (PowerShell):
python pretrain_online_trainer.py --files datafiles/*.csv --epochs 2 --max-samples 100000
"""
import argparse
import json
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":  # Executed as a script
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

# Prioritise site-packages so we import the real pandas module (not local stub)
site_packages = [p for p in sys.path if "site-packages" in p]
for sp in reversed(site_packages):
    sys.path.insert(0, sp)

import math
import random

import numpy as np
import pandas as pd
from river import forest, linear_model, metrics, preprocessing

from tradingbot_ibkr.models.online_trainer import OnlineTrainer  # noqa: E402
from tradingbot_ibkr.data import store  # noqa: E402


MACRO_CACHE = {}

BASE_FEATURE_COLUMNS = {
    'close', 'high', 'low', 'open', 'volume',
    'ret1', 'ret2', 'ret4', 'ret8',
    'ma3', 'ma6', 'ema_ratio', 'trend_slope',
    'mom5', 'mom10', 'range_pct', 'body_pct',
    'vol20', 'vol_ratio', 'vol_z', 'atr14', 'rsi14',
    'distance_high', 'distance_low',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'trend_position', 'vol_long', 'vol_quartile'
}


def _load_macro_file(path: Path) -> pd.DataFrame:
    if path in MACRO_CACHE:
        return MACRO_CACHE[path]

    df = pd.read_csv(path)
    date_col = None
    for candidate in ('date', 'DATE', 'timestamp', 'ts', 'Date'):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"Macro file {path} missing value columns")

    rename_map = {}
    if len(value_cols) == 1:
        rename_map[value_cols[0]] = path.stem
    else:
        for col in value_cols:
            rename_map[col] = f"{path.stem}_{col}"

    df.rename(columns=rename_map, inplace=True)
    df.set_index(date_col, inplace=True)
    df = df.astype(float)
    MACRO_CACHE[path] = df
    return df


def load_macro_features(macro_files):
    if not macro_files:
        return None

    frames = []
    for file in macro_files:
        df = _load_macro_file(Path(file))
        frames.append(df)

    if not frames:
        return None

    macro = pd.concat(frames, axis=1).sort_index()
    macro = macro.ffill()
    return macro


def build_examples_from_df(df, *, horizon=6, tp_pct=0.004, sl_pct=0.002,
                           max_samples=None, feature_cols=None,
                           balance=True, augment_factor=1, augment_noise=0.01,
                           rng=None):
    if len(df) <= horizon:
        return [], []

    df = df.copy()
    if rng is None:
        rng = random.Random()

    def safe(val, default=0.0):
        try:
            val = float(val)
        except Exception:
            return float(default)
        if pd.isna(val) or val != val or math.isinf(val):
            return float(default)
        return val

    # Core engineered features
    df['ret1'] = df['close'].pct_change().fillna(0.0)
    df['ret2'] = df['close'].pct_change(2).fillna(0.0)
    df['ret4'] = df['close'].pct_change(4).fillna(0.0)
    df['ret8'] = df['close'].pct_change(8).fillna(0.0)
    df['ma3'] = df['close'].rolling(3).mean().bfill()
    df['ma6'] = df['close'].rolling(6).mean().bfill()
    df['ema_fast'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema_ratio'] = ((df['ema_fast'] - df['ema_slow']) / df['close']).fillna(0.0)
    df['trend_slope'] = df['close'].diff(3).fillna(0.0)
    df['range_pct'] = ((df['high'] - df['low']) / df['close']).fillna(0.0)
    df['body_pct'] = ((df['close'] - df['open']) / df['open']).fillna(0.0)

    rolling_high = df['close'].rolling(20).max()
    rolling_low = df['close'].rolling(20).min()
    df['distance_high'] = ((df['close'] - rolling_high) / df['close']).fillna(0.0)
    df['distance_low'] = ((df['close'] - rolling_low) / df['close']).fillna(0.0)

    if 'volume' in df.columns:
        df['vol_mean20'] = df['volume'].rolling(20).mean().bfill()
        df['vol_ratio'] = df['volume'] / df['vol_mean20'].replace(0, 1)
        df['vol_z'] = ((df['volume'] - df['volume'].rolling(20).median()) /
                       df['volume'].rolling(20).std().replace(0, 1)).fillna(0.0)
    else:
        df['vol_mean20'] = 1.0
        df['vol_ratio'] = 1.0
        df['vol_z'] = 0.0

    df['vol20'] = df['ret1'].rolling(20).std().fillna(0.0)
    high_low = (df['high'] - df['low']).abs()
    high_pc = (df['high'] - df['close'].shift(1)).abs()
    low_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean().bfill()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean().replace(0, 1e-8)
    rs = roll_up / roll_down
    df['rsi14'] = 100.0 - (100.0 / (1.0 + rs))

    if isinstance(df.index, pd.DatetimeIndex):
        radians = 2 * np.pi * df.index.hour / 24
        df['hour_sin'] = np.sin(radians)
        df['hour_cos'] = np.cos(radians)
        dow = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    else:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 0.0
        df['dow_sin'] = 0.0
        df['dow_cos'] = 0.0

    trend_ema_long = df['close'].ewm(span=100, adjust=False).mean()
    df['trend_position'] = ((df['close'] - trend_ema_long) / trend_ema_long).fillna(0.0)
    df['vol_long'] = df['ret1'].rolling(100).std().fillna(0.0)
    df['vol_quartile'] = df['vol_long'].rank(pct=True).fillna(0.0)

    df['trend_ema_long'] = df['close'].ewm(span=100, adjust=False).mean()
    df['trend_position'] = ((df['close'] - df['trend_ema_long']) / df['trend_ema_long']).fillna(0.0)
    df['vol_long'] = df['ret1'].rolling(100).std().fillna(0.0)
    vol_rank = df['vol_long'].rank(pct=True).fillna(0.0)
    df['vol_quartile'] = vol_rank

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    features = []
    labels = []

    max_index = len(df) - horizon - 1 if horizon else len(df) - 1
    for idx in range(max_index):
        entry = closes[idx]
        tp_level = entry * (1 + tp_pct)
        sl_level = entry * (1 - sl_pct)
        outcome = 0
        for look_ahead in range(1, horizon + 1):
            test_idx = idx + look_ahead
            if test_idx >= len(df):
                break
            if highs[test_idx] >= tp_level:
                outcome = 1
                break
            if lows[test_idx] <= sl_level:
                outcome = 0
                break

        row = df.iloc[idx]
        if feature_cols:
            feat = {col: float(row.get(col, 0.0)) for col in feature_cols}
        else:
            feat = {
                'close': safe(row['close']),
                'high': safe(row['high']),
                'low': safe(row['low']),
                'open': safe(row.get('open', row['close'])),
                'volume': safe(row.get('volume', 0.0)),
                'ret1': safe(row['ret1']),
                'ret2': safe(row['ret2']),
                'ret4': safe(row['ret4']),
                'ret8': safe(row['ret8']),
                'ma3': safe(row['ma3']),
                'ma6': safe(row['ma6']),
                'ema_ratio': safe(row['ema_ratio']),
                'trend_slope': safe(row['trend_slope']),
                'range_pct': safe(row['range_pct']),
                'body_pct': safe(row['body_pct']),
                'vol20': safe(row['vol20']),
                'vol_ratio': safe(row['vol_ratio'], 1.0),
                'vol_z': safe(row['vol_z']),
                'atr14': safe(row['atr14']),
                'rsi14': safe(row['rsi14']),
                'distance_high': safe(row['distance_high']),
                'distance_low': safe(row['distance_low']),
                'hour_sin': safe(row['hour_sin']),
                'hour_cos': safe(row['hour_cos']),
                'dow_sin': safe(row['dow_sin']),
                'dow_cos': safe(row['dow_cos']),
                'trend_position': safe(row['trend_position']),
                'vol_long': safe(row['vol_long']),
                'vol_quartile': safe(row['vol_quartile'])
            }
            exclude_cols = BASE_FEATURE_COLUMNS | {
                'vol_mean20', 'ema_fast', 'ema_slow', 'trend_ema_long',
                'vol', 'ret', 'next_close'
            }
            for col in row.index:
                if col in feat or col in exclude_cols:
                    continue
                feat[col] = safe(row.get(col, 0.0))

        features.append(feat)
        labels.append(int(outcome))

    samples = list(zip(features, labels))

    if balance:
        positives = [s for s in samples if s[1] == 1]
        negatives = [s for s in samples if s[1] == 0]
        if positives and negatives:
            target = min(len(positives), len(negatives))
            positives = rng.sample(positives, target) if len(positives) > target else positives
            negatives = rng.sample(negatives, target) if len(negatives) > target else negatives
            samples = positives + negatives
            rng.shuffle(samples)

    if augment_factor > 1 and augment_noise > 0:
        augmented = []
        for feat, label in samples:
            augmented.append((feat, label))
            for _ in range(augment_factor - 1):
                noisy = {
                    k: v + rng.gauss(0.0, augment_noise * max(1e-6, abs(v)))
                    for k, v in feat.items()
                }
                augmented.append((noisy, label))
        samples = augmented

    if max_samples and len(samples) > max_samples:
        samples = rng.sample(samples, max_samples)

    if not samples:
        return [], []

    feats, lbls = zip(*samples)
    return list(feats), list(lbls)


def run_pretrain(paths, epochs=1, max_samples=None, threshold=0.5,
                feature_cols=None, horizon=12, profit_pct=0.01, sl_pct=None,
                balance=True, augment_factor=1, augment_noise=0.01,
                macro_files=None, model_type='logistic', job_file: str = None,
                eval_threshold: float = 0.7):
    """Programmatic pretrain entry. Writes progress/status to job_file (if provided).

    paths: list of CSV file paths
    Returns: dict with summary
    """
    trainer = OnlineTrainer()
    if model_type == 'forest':
        trainer.model = forest.ARFClassifier(seed=42, n_models=10, max_depth=8)
    else:
        trainer.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    # Fresh pretrain run starts from scratch to avoid carrying over stale weights
    trainer._samples_seen = 0
    trainer._trained = False

    total_examples = 0
    correct = 0
    seen = 0
    prob_sum = 0.0
    dist = {'pred1': 0, 'pred0': 0}
    roc_metric = metrics.ROCAUC()
    threshold_hits = 0
    threshold_wins = 0

    macro_features = load_macro_features(macro_files) if macro_files else None

    # feature_cols may be precomputed list
    for epoch in range(epochs):
        for i, path in enumerate(paths):
            df = pd.read_csv(path)
            ts_column = None
            for candidate in ('ts', 'timestamp', 'date', 'datetime'):
                if candidate in df.columns:
                    ts_column = candidate
                    break
            if ts_column is None:
                raise ValueError(f"File {path} must contain a timestamp column (ts/timestamp/date/datetime)")
            df['ts'] = pd.to_datetime(df[ts_column])
            df.set_index('ts', inplace=True)

            if macro_features is not None:
                aligned_macro = macro_features.reindex(df.index, method='ffill')
                df = df.join(aligned_macro, how='left')
                df = df.ffill().bfill()

            feats, lbls = build_examples_from_df(
                df,
                horizon=horizon,
                tp_pct=profit_pct,
                sl_pct=sl_pct or (profit_pct / 2),
                max_samples=max_samples,
                feature_cols=feature_cols,
                balance=balance,
                augment_factor=augment_factor,
                augment_noise=augment_noise,
                rng=random.Random(epoch * 101 + i)
            )

            if not feats:
                continue

            n = len(feats)
            # report starting file progress
            if job_file:
                _write_job_update(job_file, status='running', progress=0, message=f"epoch {epoch+1} file {i+1}/{len(paths)}: {n} examples")
            for idx, (feat, label) in enumerate(zip(feats, lbls)):
                prob = trainer.predict_proba(feat)
                pred = 1 if prob >= threshold else 0
                seen += 1
                prob_sum += prob
                if prob >= eval_threshold:
                    threshold_hits += 1
                    if label == 1:
                        threshold_wins += 1
                if pred == 1:
                    dist['pred1'] += 1
                else:
                    dist['pred0'] += 1
                if pred == label:
                    correct += 1
                trainer.learn_one(feat, label)
                try:
                    roc_metric.update(label, prob)
                except Exception:
                    pass
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
    trainer._samples_seen = total_examples
    trainer._trained = total_examples >= trainer._min_samples_ready
    trainer.save()
    try:
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
        'model_version': mv,
        'roc_auc': float(roc_metric.get()) if seen else None,
        'avg_probability': float(prob_sum / seen) if seen else None,
        'win_rate_at_threshold': (threshold_wins / threshold_hits) if threshold_hits else None,
        'threshold_hits': threshold_hits,
        'eval_threshold': eval_threshold
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
    parser.add_argument('--sl-pct', type=float, default=None, help='stop-loss pct used to define negative label (default profit_pct/2)')
    parser.add_argument('--no-balance', action='store_true', help='disable class balancing')
    parser.add_argument('--augment-factor', type=int, default=1, help='number of augmented copies per sample (>=1)')
    parser.add_argument('--augment-noise', type=float, default=0.01, help='relative Gaussian noise applied during augmentation')
    parser.add_argument('--macro-files', nargs='*', default=[], help='optional macro CSVs to merge (date,value) format)')
    parser.add_argument('--model-type', choices=['logistic', 'forest'], default='logistic', help='online model backbone to use during pretraining')
    parser.add_argument('--eval-threshold', type=float, default=0.7, help='probability threshold used for reporting win-rate metrics')
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
    summary = run_pretrain(
        paths,
        epochs=args.epochs,
        max_samples=args.max_samples,
        threshold=args.threshold,
        feature_cols=feature_cols,
        horizon=args.horizon,
        profit_pct=args.profit_pct,
        sl_pct=args.sl_pct,
        balance=not args.no_balance,
        augment_factor=max(1, args.augment_factor),
        augment_noise=max(0.0, args.augment_noise),
        macro_files=args.macro_files,
        model_type=args.model_type,
        job_file=args.job_file,
        eval_threshold=args.eval_threshold
    )
    print('Pretrain summary:', summary)


if __name__ == '__main__':
    main()
