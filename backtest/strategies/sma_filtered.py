from typing import Any, Optional

def _atr(df, period: int = 14):
    import pandas as pd
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean()


def generate_signals(
    df: Any,
    fast: int = 5,
    slow: int = 20,
    trend_fast: int = 50,
    trend_slow: int = 200,
    *,
    trend_ma: Optional[int] = None,
    atr_pctile: Optional[float] = None,
    atr_period: int = 14,
    atr_window: int = 200,
    cooldown: int = 0,
):
    """SMA crossover with optional trend MA, ATR percentile filter, and cooldown.

    - Base regime: SMA(fast) > SMA(slow)
    - Trend filter: either trend_fast>trend_slow or (if provided) close > SMA(trend_ma)
    - Vol filter: if atr_pctile provided, require ATR rolling percentile >= atr_pctile
    - Cooldown: force N flat bars after any 1->0 transition
    """
    import pandas as pd
    import numpy as np

    out = df.copy()
    out["sma_fast"] = out["close"].rolling(int(fast), min_periods=int(fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(slow), min_periods=int(slow)).mean()

    base = (out["sma_fast"] > out["sma_slow"]) 

    if trend_ma is not None:
        out["sma_trend"] = out["close"].rolling(int(trend_ma), min_periods=int(trend_ma)).mean()
        trend_ok = out["close"] > out["sma_trend"]
    else:
        out["trend_fast"] = out["close"].rolling(int(trend_fast), min_periods=int(trend_fast)).mean()
        out["trend_slow"] = out["close"].rolling(int(trend_slow), min_periods=int(trend_slow)).mean()
        trend_ok = out["trend_fast"] > out["trend_slow"]

    sig = (base & trend_ok)

    if atr_pctile is not None:
        atr = _atr(out, int(atr_period))
        atr_rank = atr.rolling(int(atr_window), min_periods=int(atr_window)).rank(pct=True)
        sig = sig & (atr_rank >= float(atr_pctile))

    sig = sig.astype(int)

    if int(cooldown) > 0:
        prev = sig.shift(1).fillna(0).astype(int)
        exits = (prev == 1) & (sig == 0)
        mask = np.zeros(len(sig), dtype=bool)
        last_exit = -1
        for i, ex in enumerate(exits):
            if ex:
                last_exit = i
            if last_exit >= 0 and i <= last_exit + int(cooldown):
                mask[i] = True
        sig[mask] = 0

    out["signals"] = sig
    return out[["signals"]]
