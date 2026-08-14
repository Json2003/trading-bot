"""Causal volatility, jump, and liquidity metrics for research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(close: pd.Series, window: int = 24, annualization: int = 365 * 24) -> pd.Series:
    returns = close.astype(float).pct_change()
    return returns.rolling(window).std() * float(annualization) ** 0.5


def atr_percentile(atr: pd.Series, window: int = 240) -> pd.Series:
    """Percentile of current ATR using only observations through the prior bar."""

    return atr.rolling(window, min_periods=max(10, window // 4)).rank(pct=True).shift(1)


def volatility_of_volatility(realized: pd.Series, window: int = 24) -> pd.Series:
    return realized.astype(float).rolling(window).std()


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
    value = (np.log(high.astype(float) / low.astype(float)) ** 2) / (4.0 * np.log(2.0))
    return value.rolling(window).mean() ** 0.5


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24
) -> pd.Series:
    log_hl = np.log(high.astype(float) / low.astype(float)) ** 2
    log_co = np.log(close.astype(float) / open_.astype(float)) ** 2
    value = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return value.clip(lower=0).rolling(window).mean() ** 0.5


def jump_score(close: pd.Series, window: int = 24) -> pd.Series:
    returns = close.astype(float).pct_change()
    median = returns.rolling(window).median()
    mad = (returns - median).abs().rolling(window).median()
    return (returns - median).abs() / (1.4826 * mad.replace(0, np.nan))


def volume_zscore(volume: pd.Series, window: int = 24) -> pd.Series:
    mean = volume.astype(float).rolling(window).mean()
    std = volume.astype(float).rolling(window).std().replace(0, np.nan)
    return (volume.astype(float) - mean) / std


def amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 24) -> pd.Series:
    dollar_volume = (close.astype(float) * volume.astype(float)).replace(0, np.nan)
    return (close.astype(float).pct_change().abs() / dollar_volume).rolling(window).mean()


def volatility_features(frame: pd.DataFrame, *, window: int = 24) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing volatility columns: {sorted(missing)}")
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    atr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1).rolling(14).mean()
    realized = realized_volatility(close, window)
    return pd.DataFrame(
        {
            "realized_volatility": realized,
            "atr": atr,
            "atr_percentile": atr_percentile(atr, max(window * 10, 40)),
            "volatility_of_volatility": volatility_of_volatility(realized, window),
            "parkinson_volatility": parkinson_volatility(high, low, window),
            "garman_klass_volatility": garman_klass_volatility(open_, high, low, close, window),
            "jump_score": jump_score(close, window),
            "volume_zscore": volume_zscore(frame["volume"], window),
            "amihud_illiquidity": amihud_illiquidity(close, frame["volume"], window),
        },
        index=frame.index,
    )


__all__ = [
    "amihud_illiquidity",
    "atr_percentile",
    "garman_klass_volatility",
    "jump_score",
    "parkinson_volatility",
    "realized_volatility",
    "volatility_features",
    "volatility_of_volatility",
    "volume_zscore",
]
