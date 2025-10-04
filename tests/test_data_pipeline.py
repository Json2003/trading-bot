import pandas as pd
from data_pipeline import (
    canonicalize_ohlcv,
    directional_return_label,
    drop_anomalies,
)


def test_drop_anomalies():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024", periods=3, freq="T"),
            "open": [1, 1, 1],
            "high": [2, 0, 2],
            "low": [0, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, -1, 1],
        }
    )
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


def test_canonicalize_ohlcv_respects_session_boundaries():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 09:30:00",
                "2024-01-01 09:31:00",
                "2024-01-02 09:30:00",
            ],
            "open": [1.0, 1.1, 2.0],
            "high": [1.2, 1.3, 2.2],
            "low": [0.9, 1.0, 1.9],
            "close": [1.05, 1.15, 2.05],
            "volume": [100, 120, 150],
        }
    )

    out = canonicalize_ohlcv(df, "1min", session_tz="America/New_York")

    assert len(out) == 3
    # Ensure there is no forward fill bridging across different sessions
    first_session_end = out.iloc[1]["timestamp"]
    second_session_start = out.iloc[2]["timestamp"]
    assert (second_session_start - first_session_end).total_seconds() > 3600
