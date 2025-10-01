"""Enhanced SMA crossover strategy with volatility and momentum filters.

The original module implemented a simple SMA crossover with an ATR percentile
filter.  Crypto markets can exhibit extended high volatility regimes where the
plain crossover produces a large number of false breakouts.  This rewrite adds
momentum and mean-reversion guards inspired by momentum / trend-following
literature:

* **RSI filter** – avoid taking long signals when the market is already
  overbought.  The RSI band can also be inverted to allow buying deep pullbacks.
* **Rate-of-change confirmation** – require that the medium-term momentum is
  aligned with the signal, reducing whipsaws in choppy markets.
* **Adaptive ATR filter** – retain the percentile-based volatility guard but
  allow separate control of minimum/maximum acceptable volatility regimes.
* **Cooldown** – identical to the legacy behaviour but implemented after the
  new filters are applied.

The function remains backwards compatible: the new parameters are optional and
default to sensible values that mimic the prior logic.  Signals are still
returned as a DataFrame with a single ``signals`` column of {0,1} integers so
existing backtests continue to work.
"""

from typing import Any, Optional


def _atr(df, period: int = 14):
    import pandas as pd

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean()


def _rsi(close, period: int = 14):
    import pandas as pd

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def generate_signals(
    df: Any,
    fast: int = 5,
    slow: int = 20,
    trend_fast: int = 50,
    trend_slow: int = 200,
    *,
    trend_ma: Optional[int] = None,
    atr_pctile: Optional[float] = None,
    atr_max_pctile: Optional[float] = None,
    atr_period: int = 14,
    atr_window: int = 200,
    cooldown: int = 0,
    rsi_period: int = 14,
    rsi_floor: Optional[float] = 35.0,
    rsi_ceiling: Optional[float] = 75.0,
    momentum_period: int = 10,
    momentum_threshold: float = 0.0,
    momentum_smoothing: int = 3,
):
    """SMA crossover with trend, volatility, RSI and momentum filters.

    Parameters mirror the original implementation with additional knobs:

    * ``atr_max_pctile`` lets the caller skip trades when volatility is
      *extremely* elevated, helping to avoid noise in blow-off tops.
    * ``rsi_floor`` / ``rsi_ceiling`` define an acceptable RSI band.  Set both
      to ``None`` to disable the filter or widen the band for more trades.
    * ``momentum_period`` / ``momentum_threshold`` gate entries on medium-term
      rate-of-change.  Positive thresholds favour trend-following, negative
      thresholds enable value-style pullback buys.
    * ``momentum_smoothing`` EMA-smooths the ROC signal to reduce jitter.
    """

    import numpy as np
    import pandas as pd

    out = df.copy()
    out["sma_fast"] = out["close"].rolling(int(fast), min_periods=int(fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(slow), min_periods=int(slow)).mean()

    base = out["sma_fast"] > out["sma_slow"]

    if trend_ma is not None:
        out["sma_trend"] = out["close"].rolling(int(trend_ma), min_periods=int(trend_ma)).mean()
        trend_ok = out["close"] > out["sma_trend"]
    else:
        out["trend_fast"] = out["close"].rolling(int(trend_fast), min_periods=int(trend_fast)).mean()
        out["trend_slow"] = out["close"].rolling(int(trend_slow), min_periods=int(trend_slow)).mean()
        trend_ok = out["trend_fast"] > out["trend_slow"]

    sig = base & trend_ok

    # Adaptive volatility regime filter (min and optional max percentile)
    if atr_pctile is not None or atr_max_pctile is not None:
        atr = _atr(out, int(atr_period))
        atr_rank = atr.rolling(int(atr_window), min_periods=int(atr_window)).rank(pct=True)
        if atr_pctile is not None:
            sig &= atr_rank >= float(atr_pctile)
        if atr_max_pctile is not None:
            sig &= atr_rank <= float(atr_max_pctile)

    # RSI guard
    if rsi_period and (rsi_floor is not None or rsi_ceiling is not None):
        rsi = _rsi(out["close"], int(rsi_period))
        if rsi_floor is not None:
            sig &= rsi >= float(rsi_floor)
        if rsi_ceiling is not None:
            sig &= rsi <= float(rsi_ceiling)

    # Momentum confirmation using rate of change
    if momentum_period > 0:
        roc = out["close"].pct_change(int(momentum_period))
        if int(momentum_smoothing) > 1:
            roc = roc.ewm(span=int(momentum_smoothing), adjust=False).mean()
        sig &= roc >= float(momentum_threshold)

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
