import pandas as pd
import numpy as np


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Simplified ADX (Wilder smoothing not fully implemented)
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(int(period), min_periods=int(period)).mean()
    pdi = 100 * (plus_dm.rolling(int(period), min_periods=int(period)).mean() / atr).replace([np.inf, -np.inf], np.nan)
    mdi = 100 * (minus_dm.rolling(int(period), min_periods=int(period)).mean() / atr).replace([np.inf, -np.inf], np.nan)
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.rolling(int(period), min_periods=int(period)).mean()


def generate_signals(
    df: pd.DataFrame,
    fast: int = 8,
    slow: int = 21,
    trend_ma: int = 200,
    atr_period: int = 14,
    adx_period: int = 14,
    adx_min: float = 18.0,
    slope_window: int = 50,
    slope_min_bull: float = 0.0,
    slope_max_bear: float = 0.0,
    atr_window: int = 200,
    atr_pctile_bull: float = 0.30,
    atr_pctile_bear: float = 0.40,
    cooldown: int = 3,
    enable_shorts: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(int(fast), min_periods=int(fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(slow), min_periods=int(slow)).mean()
    out["sma_trend"] = out["close"].rolling(int(trend_ma), min_periods=int(trend_ma)).mean()

    # Trend slope
    out["sma_trend_slope"] = out["sma_trend"] - out["sma_trend"].shift(int(slope_window))

    atr = _atr(out, int(atr_period))
    adx = _adx(out, int(adx_period))

    # ATR rolling percentile
    atr_roll = atr.rolling(int(atr_window), min_periods=int(atr_window))
    atr_pct = atr_roll.rank(pct=True)

    # Long regime
    long_base = (
        (out["sma_fast"] > out["sma_slow"]) &
        (out["close"] > out["sma_trend"]) &
        (out["sma_trend_slope"] > float(slope_min_bull)) &
        (adx >= float(adx_min)) &
        (atr_pct >= float(atr_pctile_bull))
    )

    # Short regime
    short_base = (
        bool(enable_shorts) &
        (out["sma_fast"] < out["sma_slow"]) &
        (out["close"] < out["sma_trend"]) &
        (out["sma_trend_slope"] < float(slope_max_bear)) &
        (adx >= float(adx_min)) &
        (atr_pct >= float(atr_pctile_bear))
    )

    sig = pd.Series(0, index=out.index, dtype=int)
    sig[long_base.fillna(False)] = 1
    sig[short_base.fillna(False)] = -1

    # Cooldown after exits
    if int(cooldown) > 0:
        sig_shift = sig.shift(1).fillna(0)
        exit_points = ((sig_shift != 0) & (sig == 0))
        cool_mask = np.zeros(len(sig), dtype=bool)
        last_exit = -10**9
        for i, ex in enumerate(exit_points):
            if ex:
                last_exit = i
            if i <= last_exit + int(cooldown):
                cool_mask[i] = True
        sig[cool_mask] = 0

    out["signals"] = sig
    return out[["signals"]]
