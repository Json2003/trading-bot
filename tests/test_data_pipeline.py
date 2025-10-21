import math

import pandas as pd
import numpy as np
from data_pipeline import (
    canonicalize_ohlcv,
    comp_m_scores,
    directional_return_label,
    drop_anomalies,
    magnitude_bucket_label,
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


def test_magnitude_bucket_label_handles_constant_series():
    close = pd.Series([100, 100, 100, 100])

    lbl = magnitude_bucket_label(close, horizon=1, q=4)

    assert all(value == 0 for value in lbl[:-1])
    assert math.isnan(lbl.iloc[-1])


def test_magnitude_bucket_label_handles_low_unique_bins():
    close = pd.Series([100, 101, 101, 102, 102, 103])

    lbl = magnitude_bucket_label(close, horizon=1, q=5)

    # All except the final horizon look-ahead should be labeled
    labeled = [value for value in lbl if not math.isnan(value)]
    assert len(labeled) == len(close) - 1
    assert max(labeled) <= 4


def test_comp_m_scores_computes_cross_sectional_zscores():
    price_hist = {
        "AAA": np.array([10, 10.5, 11.0, 11.5, 12.0], dtype=float),
        "BBB": np.array([8, 8.1, 8.0, 7.9, 7.8], dtype=float),
        "CCC": np.array([5, 5.0, 5.1, 5.3, 5.6], dtype=float),
        # Insufficient history should be ignored
        "DDD": np.array([1.0, 1.1], dtype=float),
    }

    scores = comp_m_scores(price_hist, lookback=4)

    assert set(scores) == {"AAA", "BBB", "CCC"}
    # Ensure values are z-scored (mean approx 0, standard deviation approx 1)
    z_values = list(scores.values())
    mean = sum(z_values) / len(z_values)
    variance = sum((value - mean) ** 2 for value in z_values) / len(z_values)
    assert abs(mean) < 1e-12
    assert abs(variance - 1.0) < 1e-12
    assert scores["AAA"] > scores["CCC"] > scores["BBB"]
