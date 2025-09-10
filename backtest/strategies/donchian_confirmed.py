from typing import Optional
import pandas as pd
import numpy as np


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean()


def generate_signals(
    df: pd.DataFrame,
    donchian_n: int = 20,
    trend_ma: Optional[int] = 200,
    adx_min: Optional[float] = None,
    adx_period: int = 14,
    atr_period: int = 14,
    atr_window: int = 200,
    atr_pctile_min: Optional[float] = None,
    cooldown: int = 0,
    enable_shorts: bool = True,
) -> pd.DataFrame:
    """Donchian channel breakout confirmation strategy.

    - Long when close breaks above the rolling N-bar high (previous bars),
      optionally also require price above trend MA and ATR percentile >= threshold.
    - Short when close breaks below the rolling N-bar low analogously.
    - Cooldown forces flat for N bars after an exit.
    """
    out = df.copy()
    n = int(donchian_n)
    # Use previous N bars for breakout reference
    out["dc_high"] = out["high"].rolling(n, min_periods=n).max().shift(1)
    out["dc_low"] = out["low"].rolling(n, min_periods=n).min().shift(1)

    if trend_ma is not None:
        tma = int(trend_ma)
        out["sma_trend"] = out["close"].rolling(tma, min_periods=tma).mean()
        trend_long_ok = out["close"] > out["sma_trend"]
        trend_short_ok = out["close"] < out["sma_trend"]
    else:
        trend_long_ok = trend_short_ok = pd.Series(True, index=out.index)

    # Optional ADX filter (simple DX as in trend_adx_atr)
    if adx_min is not None and float(adx_min) > 0:
        high, low, close = out["high"], out["low"], out["close"]
        up = high.diff(); dn = -low.diff()
        plus_dm = ((up > dn) & (up > 0)) * up
        minus_dm = ((dn > up) & (dn > 0)) * dn
        tr1 = (high - low)
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(int(adx_period), min_periods=int(adx_period)).mean()
        pdi = 100 * (plus_dm.rolling(int(adx_period), min_periods=int(adx_period)).mean() / atr).replace([np.inf, -np.inf], np.nan)
        mdi = 100 * (minus_dm.rolling(int(adx_period), min_periods=int(adx_period)).mean() / atr).replace([np.inf, -np.inf], np.nan)
        dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
        adx = dx.rolling(int(adx_period), min_periods=int(adx_period)).mean()
        adx_ok = adx >= float(adx_min)
    else:
        adx_ok = pd.Series(True, index=out.index)

    # Optional ATR percentile floor to avoid ultra-quiet conditions
    if atr_pctile_min is not None and float(atr_pctile_min) > 0:
        atr = _atr(out, int(atr_period))
        # Use lenient min_periods to avoid all-NaN when window is small
        atr_rank = atr.rolling(int(atr_window), min_periods=max(5, int(atr_window)//4)).rank(pct=True)
        vol_ok = atr_rank >= float(atr_pctile_min)
    else:
        vol_ok = pd.Series(True, index=out.index)

    long_sig = (out["close"] > out["dc_high"]) & trend_long_ok & adx_ok & vol_ok
    short_sig = (bool(enable_shorts)) & (out["close"] < out["dc_low"]) & trend_short_ok & adx_ok & vol_ok

    sig = pd.Series(0, index=out.index, dtype=int)
    sig[long_sig.fillna(False)] = 1
    sig[short_sig.fillna(False)] = -1

    if int(cooldown) > 0:
        prev = sig.shift(1).fillna(0).astype(int)
        exits = (prev != 0) & (sig == 0)
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
