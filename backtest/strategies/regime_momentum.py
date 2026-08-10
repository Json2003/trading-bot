"""Long/short momentum signals gated by trend and volatility regime."""

from __future__ import annotations

import pandas as pd


def generate_signals(
    data: pd.DataFrame,
    *,
    fast: int = 13,
    slow: int = 34,
    regime: int = 200,
    slope_bars: int = 24,
    atr_period: int = 14,
    min_atr_pct: float = 0.001,
    max_atr_pct: float = 0.08,
) -> pd.DataFrame:
    """Return {-1, 0, 1} signals using only current and prior bars."""

    if not fast >= 2 or not slow > fast or not regime > slow:
        raise ValueError("require 2 <= fast < slow < regime")
    frame = data.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_pct = true_range.rolling(atr_period).mean() / close
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    regime_ema = close.ewm(span=regime, adjust=False).mean()
    slope = regime_ema - regime_ema.shift(slope_bars)
    tradable_vol = atr_pct.between(min_atr_pct, max_atr_pct)

    signals = pd.Series(0, index=frame.index, dtype="int64")
    signals[(fast_ema > slow_ema) & (close > regime_ema) & (slope > 0) & tradable_vol] = 1
    signals[(fast_ema < slow_ema) & (close < regime_ema) & (slope < 0) & tradable_vol] = -1
    return pd.DataFrame({"signals": signals})


__all__ = ["generate_signals"]
