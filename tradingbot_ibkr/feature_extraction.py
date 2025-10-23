"""Feature extraction utilities for trading signals.

The module now combines technical market structure, fundamental context, and
microstructure order-flow signals so that downstream models receive a rich
feature matrix.  Lightweight fallbacks are used when optional data (e.g.
earnings releases) are not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def technical_indicators(df: pd.DataFrame, *, rsi_period: int = 14) -> pd.DataFrame:
    """Compute core technical indicators used by the ML stack.

    Parameters
    ----------
    df:
        DataFrame with at least ``open``, ``high``, ``low``, ``close`` and
        optionally ``volume`` columns.
    rsi_period:
        Number of periods for the RSI calculation.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional indicator columns such as moving
        averages, momentum, and volatility measures.
    """

    if "close" not in df:
        raise KeyError("technical_indicators requires a 'close' column")

    out = df.copy()

    # Moving averages for mean-reversion vs trend context
    out["ma_fast"] = out["close"].rolling(window=10, min_periods=1).mean()
    out["ma_slow"] = out["close"].rolling(window=30, min_periods=1).mean()

    # Short-term momentum windows reused by the backtester
    out["ret1"] = out["close"].pct_change().fillna(0.0)
    out["ma3"] = out["close"].rolling(3, min_periods=1).mean()
    out["mom5"] = out["close"].pct_change(5).fillna(0.0)
    out["mom10"] = out["close"].pct_change(10).fillna(0.0)

    # Average True Range / volatility proxies
    high_low = out["high"] - out["low"]
    high_close_prev = (out["high"] - out["close"].shift(1)).abs()
    low_close_prev = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)
    out["vol20"] = out["ret1"].rolling(20, min_periods=1).std().fillna(0.0)

    if "volume" in out:
        out["vol_mean20"] = out["volume"].rolling(20, min_periods=1).mean().replace(0, np.nan)
        out["vol_ratio"] = (out["volume"] / out["vol_mean20"]).fillna(0.0)
    else:
        out["vol_mean20"] = 0.0
        out["vol_ratio"] = 0.0

    # RSI with numerically-stable operations
    delta = out["close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=rsi_period, adjust=False).mean()
    roll_down = down.ewm(span=rsi_period, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi14"] = out["rsi14"].bfill().fillna(50.0)

    return out


def fundamental_indicators(
    df: pd.DataFrame,
    earnings: Optional[pd.DataFrame] = None,
    *,
    trailing_pe_floor: float = 2.0,
    trailing_pe_cap: float = 80.0,
) -> pd.DataFrame:
    """Create valuation and earnings-derived features.

    The helper tolerates missing fundamental data by constructing
    economically-reasonable proxies from price history when actual EPS
    observations are absent.  These features are consumed both by ML models
    and the fundamental guard-rails that gate live trades.
    """

    if "close" not in df:
        raise KeyError("fundamental_indicators requires a 'close' column")

    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)

    if earnings is not None and not earnings.empty:
        aligned = earnings.reindex(df.index, method="ffill")
        eps_actual_aligned = (
            aligned["eps_actual"] if "eps_actual" in aligned.columns else pd.Series(np.nan, index=df.index)
        )
        eps_est_aligned = (
            aligned["eps_estimate"] if "eps_estimate" in aligned.columns else pd.Series(0.0, index=df.index)
        )
        trailing_eps = eps_actual_aligned
        surprise = (eps_actual_aligned - eps_est_aligned).fillna(0.0)
        trailing_eps = trailing_eps.replace(0, np.nan)
    else:
        # Proxy EPS using a conservative earnings yield derived from long-term
        # price averages; avoids division by zero and keeps the filter stable.
        trailing_eps = (close.rolling(252, min_periods=30).mean() / 25.0).replace(0, np.nan)
        surprise = close.pct_change(90).fillna(0.0)

    pe_ratio = (close / trailing_eps).clip(lower=trailing_pe_floor, upper=trailing_pe_cap)
    earnings_yield = (1.0 / pe_ratio).replace([np.inf, -np.inf], np.nan)
    earnings_growth = close.pct_change(252).fillna(0.0)
    earnings_surprise = surprise

    # Composite score blending valuation, growth, and surprise.
    fundamental_score = (
        0.4 * earnings_growth.fillna(0.0)
        + 0.3 * earnings_surprise.fillna(0.0)
        + 0.3 * earnings_yield.fillna(0.0)
    )

    out["pe_ratio"] = pe_ratio.ffill().fillna(pe_ratio.median())
    out["earnings_yield"] = earnings_yield.ffill().fillna(0.0)
    out["earnings_growth"] = earnings_growth.fillna(0.0)
    out["earnings_surprise"] = earnings_surprise.fillna(0.0)
    out["fundamental_score"] = fundamental_score.fillna(0.0)

    return out


def news_sentiment(_: pd.DataFrame) -> pd.Series:
    """Placeholder for news sentiment extraction.

    Returns a neutral sentiment score for each row.
    """

    return pd.Series(0.0)


def orderbook_features(_: pd.DataFrame) -> pd.Series:
    """Backward compatible alias kept for legacy imports."""

    return pd.Series(0.0)


def orderflow_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Approximate limit-order-book style features from price/volume bars.

    Parameters
    ----------
    df:
        Input dataframe containing at least ``close`` and, when available,
        ``volume``.
    window:
        Lookback window for smoothing the order-flow statistics.
    """

    vol = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)
    price_change = df["close"].diff().fillna(0.0)
    signed_volume = vol * np.sign(price_change)

    out = pd.DataFrame(index=df.index)
    out["order_imbalance"] = signed_volume.rolling(window, min_periods=1).sum().fillna(0.0)

    denom = vol.rolling(window, min_periods=1).sum().replace(0, np.nan)
    out["flow_velocity"] = (out["order_imbalance"] / denom).fillna(0.0)

    mean_vol = vol.rolling(window, min_periods=1).mean().replace(0, np.nan)
    out["liquidity_ratio"] = (vol / mean_vol).fillna(0.0)
    out["order_flow_trend"] = signed_volume.rolling(window, min_periods=1).mean().fillna(0.0)

    return out


@dataclass
class FeatureBundle:
    """Convenience container with pre-computed feature subsets."""

    technical: pd.DataFrame
    fundamentals: pd.DataFrame
    orderflow: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate the stored frames with proper column alignment."""

        combined = self.technical.copy()
        for frame in (self.fundamentals, self.orderflow):
            for column in frame.columns:
                combined[column] = frame[column]
        return combined


def build_feature_matrix(
    df: pd.DataFrame, earnings: Optional[pd.DataFrame] = None
) -> FeatureBundle:
    """Return a :class:`FeatureBundle` combining technical, fundamental, and order-flow features."""

    technical = technical_indicators(df)
    fundamentals = fundamental_indicators(df, earnings)
    orderflow = orderflow_features(df)
    return FeatureBundle(technical=technical, fundamentals=fundamentals, orderflow=orderflow)
