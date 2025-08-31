"""Batch trainer: simple scikit-learn training and walk-forward evaluation.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import logging
from typing import Tuple

MODEL_DIR = Path(__file__).resolve().parents[1] / 'model_store'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def featurize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # create simple features and a binary target of next-bar up/down
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df.dropna(inplace=True)
    X = df[['ret','vol_change','ma5','ma10']]
    y = (df['close'].shift(-1) > df['close']).astype(int)[:-1]
    X = X[:-1]
    return X, y

def train_and_evaluate(bars_df: pd.DataFrame):
    X, y = featurize(bars_df)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
    avg_score = sum(scores)/len(scores) if scores else 0.0
    model_path = MODEL_DIR / 'batch_model.joblib'
    joblib.dump(clf, model_path)
    return {'cv_scores': scores, 'avg_score': avg_score, 'model_path': str(model_path)}
