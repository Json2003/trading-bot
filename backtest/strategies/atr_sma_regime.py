from typing import Any

def _atr(df, period: int = 14):
    import pandas as pd
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


essential_params = [
    "fast", "slow", "trend_ma", "atr_period", "atr_pctile", "atr_window", "cooldown"
]

def generate_signals(
    df: Any,
    fast: int = 10,
    slow: int = 30,
    trend_ma: int = 200,
    atr_period: int = 14,
    atr_pctile: float = 0.4,
    atr_window: int = 200,
    cooldown: int = 3,
):
    """Long-only SMA cross with regime/volatility filters and cooldown.

    Entry: SMA(fast)>SMA(slow), close>SMA(trend_ma), ATR percentile >= atr_pctile.
    Exit handled by engine (flip/TP/SL). Output is 1 or 0 regime.
    """
    import pandas as pd
    import numpy as np

    out = df.copy()
    out["sma_fast"] = out["close"].rolling(int(fast), min_periods=int(fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(slow), min_periods=int(slow)).mean()
    out["sma_trend"] = out["close"].rolling(int(trend_ma), min_periods=int(trend_ma)).mean()

    atr = _atr(out, int(atr_period))
    atr_roll = atr.rolling(int(atr_window), min_periods=int(atr_window))
    atr_rank = atr_roll.rank(pct=True)
    out["atr_pct"] = atr_rank

    sig = (
        (out["sma_fast"] > out["sma_slow"]) &
        (out["close"] > out["sma_trend"]) &
        (out["atr_pct"] >= float(atr_pctile))
    ).astype(int)

    sig_shift = sig.shift(1).fillna(0).astype(int)
    exit_points = (sig_shift == 1) & (sig == 0)

    cooldown_mask = np.zeros(len(out), dtype=bool)
    last_exit_idx = -1
    for i, ex in enumerate(exit_points):
        if ex:
            last_exit_idx = i
        if last_exit_idx >= 0 and i <= last_exit_idx + int(cooldown):
            cooldown_mask[i] = True

    sig[cooldown_mask] = 0
    out["signals"] = sig
    return out[["signals"]]
