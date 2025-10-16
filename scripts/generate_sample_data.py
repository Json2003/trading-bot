import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01 00:00:00", periods=200, freq="T")
base_price = 100
prices = base_price + np.cumsum(np.random.normal(0, 0.5, 200))
df = pd.DataFrame({
    "timestamp": dates,
    "open": prices,
    "high": prices + np.random.uniform(0, 0.5, 200),
    "low": prices - np.random.uniform(0, 0.5, 200),
    "close": prices + np.random.uniform(-0.2, 0.2, 200),
    "volume": np.random.randint(500, 1500, 200)
})
df.to_csv("backtest/sample_data/sample_ohlcv.csv", index=False)
