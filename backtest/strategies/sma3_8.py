# User-provided SMA crossover (3 over 8)
# Keep pandas import local to function to avoid repo-local stubs affecting module import.
from typing import Any

def generate_signals(df: Any, fast: int = 3, slow: int = 8):
    import pandas as pd  # local import ensures real pandas when called from CLI
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(int(fast), min_periods=int(fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(slow), min_periods=int(slow)).mean()
    out["signals"] = 0
    out.loc[out["sma_fast"] > out["sma_slow"], "signals"] = 1
    return out[["signals"]]
