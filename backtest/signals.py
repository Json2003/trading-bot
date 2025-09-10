"""Simple signal generators for backtesting.

Implements a basic SMA crossover signal generator.
Avoids importing pandas at module import time to prevent shadowing by
repo-local files named pandas.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only for type hints; won't import at runtime
    import pandas as pd  # type: ignore


def generate_signals(df):
    """Return a DataFrame with a single 'signals' column (0/1) based on SMA(20/60).

    Rules:
    - signals == 1 when SMA(20) > SMA(60)
    - otherwise 0

    Expects input df to have a 'close' column and support pandas-like ops.
    """
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(20, min_periods=20).mean()
    out["sma_slow"] = out["close"].rolling(60, min_periods=60).mean()
    out["signals"] = 0
    out.loc[out["sma_fast"] > out["sma_slow"], "signals"] = 1
    return out[["signals"]]
