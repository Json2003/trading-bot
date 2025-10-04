import pandas as pd
from data_pipeline import (
    canonicalize_ohlcv,
    directional_return_label,
    drop_anomalies,
)


def test_drop_anomalies():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024", periods=3, freq="T"),
        "open": [1, 1, 1],
        "high": [2, 0, 2],
        "low": [0, 1, 1],
        "close": [1, 1, 1],
        "volume": [1, -1, 1],
    })
    out = drop_anomalies(df)
    assert len(out) == 1


def test_directional_return_label():
    s = pd.Series([1, 2, 1])
    lbl = directional_return_label(s, 1)
    assert list(lbl)[:2] == [1, -1]


def test_canonicalize_ohlcv_localizes_naive_timestamps():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 09:30", periods=2, freq="1min"),
        "open": [1.0, 1.1],
        "high": [1.2, 1.3],
        "low": [0.9, 1.0],
        "close": [1.05, 1.15],
        "volume": [100, 120],
    })

    out = canonicalize_ohlcv(df, "1min", session_tz="America/New_York")

    expected_index = pd.date_range(
        "2024-01-01 14:30", periods=2, freq="1min", tz="UTC"
    )
    assert list(out["timestamp"]) == list(expected_index)
