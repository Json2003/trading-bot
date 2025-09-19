import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

FEAT = "data/parquet/features_1m"
LAB = "data/parquet/labels_1m"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def load_join() -> pl.DataFrame:
    dfs = []
    for sym in SYMS:
        features = pl.scan_parquet(f"{FEAT}/symbol={sym}/date=*/part.parquet")
        labels = pl.scan_parquet(f"{LAB}/symbol={sym}/labels.parquet")
        df = (
            features.join(labels, on=["symbol", "ts"], how="inner")
            .select(
                [
                    "symbol",
                    "ts",
                    "close",
                    "ret_5m",
                    "ret_30m",
                    "vol_realized_30m",
                    "zscore_30m",
                    "rsi_14",
                    "y",
                ]
            )
        )
        dfs.append(df)
    return pl.concat(dfs).sort(["ts", "symbol"]).collect()


if __name__ == "__main__":
    df = load_join().drop_nulls()
    X = df.drop(["symbol", "ts", "y"]).to_numpy()
    y = (df["y"] == 1).to_numpy().astype(np.uint8)  # classify "up" events
    tscv = TimeSeriesSplit(n_splits=5)

    aps, f1s = [], []
    for tr, va in tscv.split(X):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=64,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_data_in_leaf=50,
            metric="auc",
        )
        model = lgb.train(
            params, dtr, num_boost_round=200, valid_sets=[dva], verbose_eval=False
        )
        preds = model.predict(X[va])
        aps.append(average_precision_score(y[va], preds))
        f1s.append(f1_score(y[va], (preds > 0.6).astype(int)))
    os.makedirs("artifacts/models", exist_ok=True)
    ts = int(time.time())
    model.save_model(f"artifacts/models/lgbm_{ts}.txt")
    with open(f"artifacts/models/metrics_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"AP_mean": float(np.mean(aps)), "F1_mean": float(np.mean(f1s))}, f, indent=2)
    print("Saved model with AP=", np.mean(aps), "F1=", np.mean(f1s))
