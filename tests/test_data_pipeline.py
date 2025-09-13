import pandas as pd
from data_pipeline import drop_anomalies, directional_return_label


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
