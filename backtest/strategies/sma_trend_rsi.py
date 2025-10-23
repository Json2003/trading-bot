"""SMA crossover strategy with trend and RSI confirmation filters."""

from __future__ import annotations

from typing import Any, Optional


def _rsi(close, period: int) -> "pd.Series":
    import pandas as pd

    if period <= 0:
        return pd.Series(float("nan"), index=close.index)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    span = max(1, 2 * period - 1)
    avg_gain = gain.ewm(span=span, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=span, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def generate_signals(
    df: Any,
    fast: int = 8,
    slow: int = 34,
    trend_fast: int = 55,
    trend_slow: int = 144,
    rsi_period: int = 14,
    rsi_floor: Optional[float] = 30.0,
    rsi_ceiling: Optional[float] = 70.0,
) -> "pd.DataFrame":
    """Generate long/exit signals with SMA crossovers and RSI confirmation."""

    import numpy as np
    import pandas as pd

    out = df.copy()

    out["sma_fast"] = out["close"].rolling(int(fast), min_periods=int(fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(slow), min_periods=int(slow)).mean()
    out["trend_fast"] = out["close"].rolling(int(trend_fast), min_periods=int(trend_fast)).mean()
    out["trend_slow"] = out["close"].rolling(int(trend_slow), min_periods=int(trend_slow)).mean()

    rsi = _rsi(out["close"], int(rsi_period)) if rsi_period else pd.Series(np.nan, index=out.index)

    crossover_up = (out["sma_fast"] > out["sma_slow"]) & (
        out["sma_fast"].shift(1) <= out["sma_slow"].shift(1)
    )
    crossover_down = (out["sma_fast"] < out["sma_slow"]) & (
        out["sma_fast"].shift(1) >= out["sma_slow"].shift(1)
    )

    trend_ok = out["trend_fast"] > out["trend_slow"]

    if rsi_period and (rsi_floor is not None or rsi_ceiling is not None):
        rsi_filter = pd.Series(True, index=out.index)
        if rsi_floor is not None:
            rsi_filter &= rsi > float(rsi_floor)
        if rsi_ceiling is not None:
            rsi_filter &= rsi < float(rsi_ceiling)
    else:
        rsi_filter = pd.Series(True, index=out.index)

    long_signals = crossover_up & trend_ok & rsi_filter

    exit_signals = crossover_down.copy()
    if rsi_period and rsi_ceiling is not None:
        exit_signals |= rsi > float(rsi_ceiling)

    signals = pd.Series(0, index=out.index, dtype=int)
    signals[long_signals.fillna(False)] = 1
    signals[exit_signals.fillna(False)] = -1

    out["signals"] = signals
    return out[["signals"]]
