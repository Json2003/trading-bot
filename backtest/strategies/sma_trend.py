"""Basic SMA crossover strategy with trend confirmation."""

from __future__ import annotations

from typing import Any


def _sma(close, window: int):
    """Return the simple moving average for ``close`` with full periods only."""

    if window <= 0:
        raise ValueError("window must be positive")

    return close.rolling(int(window), min_periods=int(window)).mean()


def generate_signals(
    df: Any,
    fast: int = 8,
    slow: int = 34,
    trend_fast: int = 55,
    trend_slow: int = 144,
):
    """Generate buy/sell signals based on SMA crossovers with a trend filter."""

    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    out = df.copy()

    close = out["close"].astype(float)
    out["sma_fast"] = _sma(close, int(fast))
    out["sma_slow"] = _sma(close, int(slow))
    out["trend_fast"] = _sma(close, int(trend_fast))
    out["trend_slow"] = _sma(close, int(trend_slow))

    long_mask = (out["sma_fast"] > out["sma_slow"]) & (
        out["trend_fast"] > out["trend_slow"]
    )
    short_mask = (out["sma_fast"] < out["sma_slow"]) & (
        out["trend_fast"] < out["trend_slow"]
    )

    long_mask = long_mask.fillna(False)
    short_mask = short_mask.fillna(False)

    signal_values = []
    for long_flag, short_flag in zip(long_mask.values, short_mask.values):
        if bool(long_flag):
            signal_values.append(1)
        elif bool(short_flag):
            signal_values.append(-1)
        else:
            signal_values.append(0)

    out["signal"] = signal_values
    return out


__all__ = ["generate_signals"]
