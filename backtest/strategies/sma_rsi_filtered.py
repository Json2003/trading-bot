"""SMA crossover strategy with an EMA trend filter and RSI guard."""

from __future__ import annotations

from typing import Any, Optional


def _ema(series, length: int):
    import pandas as pd

    length = int(length)
    if length <= 0:
        return pd.Series(float("nan"), index=series.index)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rsi(close, period: int) -> "pd.Series":
    import pandas as pd

    period = int(period)
    if period <= 0:
        return pd.Series(float("nan"), index=close.index)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def generate_signals(
    df: Any,
    fast: int = 8,
    slow: int = 34,
    trend_fast: int = 55,
    trend_slow: int = 144,
    rsi_period: int = 14,
    rsi_floor: Optional[float] = 30.0,
    rsi_ceiling: Optional[float] = 70.0,
    cooldown: int = 0,
) -> "pd.DataFrame":
    """Return a ``signals`` column with {1, 0, -1} trade instructions.

    The strategy goes long when the fast SMA crosses above the slow SMA while
    price is aligned with a slower trend filter and the RSI lies within the
    configured band.  Positions are closed when the fast SMA crosses back below
    the slow SMA (or the RSI exceeds the ceiling).  ``cooldown`` optionally
    enforces a flat period after exits to avoid immediate re-entries.
    """

    import numpy as np
    import pandas as pd

    data = pd.DataFrame(df).copy()
    for col in ("open", "high", "low", "close"):
        if col not in data:
            raise ValueError("DataFrame must include OHLC columns")

    fast = max(int(fast), 1)
    slow = max(int(slow), 1)
    trend_fast = max(int(trend_fast), 1)
    trend_slow = max(int(trend_slow), 1)

    data["sma_fast"] = data["close"].rolling(fast, min_periods=fast).mean()
    data["sma_slow"] = data["close"].rolling(slow, min_periods=slow).mean()
    data["trend_fast"] = _ema(data["close"], trend_fast)
    data["trend_slow"] = _ema(data["close"], trend_slow)

    crossover_up = (data["sma_fast"] > data["sma_slow"]) & (
        data["sma_fast"].shift(1) <= data["sma_slow"].shift(1)
    )
    crossover_down = (data["sma_fast"] < data["sma_slow"]) & (
        data["sma_fast"].shift(1) >= data["sma_slow"].shift(1)
    )

    trend_ok = data["trend_fast"] > data["trend_slow"]

    if rsi_period and (rsi_floor is not None or rsi_ceiling is not None):
        rsi = _rsi(data["close"], rsi_period)
        rsi_mask = pd.Series(True, index=data.index)
        if rsi_floor is not None:
            rsi_mask &= rsi >= float(rsi_floor)
        if rsi_ceiling is not None:
            rsi_mask &= rsi <= float(rsi_ceiling)
    else:
        rsi_mask = pd.Series(True, index=data.index)
        rsi = pd.Series(np.nan, index=data.index)

    entries = crossover_up & trend_ok & rsi_mask

    exits = crossover_down.copy()
    if rsi_period and rsi_ceiling is not None:
        exits |= rsi > float(rsi_ceiling)

    signals = pd.Series(0, index=data.index, dtype=int)
    signals[entries.fillna(False)] = 1
    signals[exits.fillna(False)] = -1

    if cooldown:
        cooldown = int(max(cooldown, 0))
        last_exit = -np.inf
        for idx, is_exit in enumerate(exits.fillna(False)):
            if is_exit:
                last_exit = idx
                continue
            if idx <= last_exit + cooldown:
                signals.iloc[idx] = 0

    return pd.DataFrame({"signals": signals})
