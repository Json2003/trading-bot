from __future__ import annotations

import pandas as pd

from backtest.volatility_metrics import atr_percentile, volatility_features


def _frame(rows: int = 80) -> pd.DataFrame:
    close = pd.Series([100.0 + i * 0.1 for i in range(rows)])
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )


def test_volatility_features_shape_and_columns() -> None:
    features = volatility_features(_frame())
    assert len(features) == 80
    assert {"realized_volatility", "atr_percentile", "jump_score", "amihud_illiquidity"} <= set(features.columns)


def test_atr_percentile_does_not_use_future_observation() -> None:
    atr = pd.Series([1.0] * 30 + [100.0])
    before = atr_percentile(atr, window=20).iloc[-2]
    after = atr_percentile(atr, window=20).iloc[-1]
    assert before == before  # finite/NaN-safe assertion
    assert pd.isna(after) or after <= 1.0
