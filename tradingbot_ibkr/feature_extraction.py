"""Feature extraction utilities for trading signals.

Provides technical indicators, news sentiment, and order book features.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple technical indicators.

    Currently calculates fast/slow moving averages and RSI.

    Args:
        df: DataFrame with at least a ``close`` column.

    Returns:
        DataFrame with added indicator columns.
    """
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_slow"] = df["close"].rolling(window=30, min_periods=1).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def news_sentiment(_: pd.DataFrame) -> pd.Series:
    """Placeholder for news sentiment extraction.

    Returns a neutral sentiment score for each row.
    """
    return pd.Series(0.0)


def orderbook_features(_: pd.DataFrame) -> pd.Series:
    """Placeholder for limit order book features.

    Returns zeros until integrated with real order book data.
    """
    return pd.Series(0.0)
