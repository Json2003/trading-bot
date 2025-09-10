import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
for shadow in ("pandas", "requests", "ccxt"):
    if shadow in sys.modules:
        del sys.modules[shadow]
site_paths = [p for p in sys.path if "site-packages" in p]
non_site = [p for p in sys.path if "site-packages" not in p]
sys.path[:] = site_paths + non_site

import pandas as pd
from backtest.strategies.donchian_confirmed import generate_signals


def test_breakout_logic():
    ts = pd.date_range("2024-01-01", periods=30, freq="H", tz="UTC")
    close = pd.Series([100]*10 + [110]*10 + [120]*10, index=ts)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1,
    })
    out = generate_signals(df, donchian_n=5, trend_ma=10, adx_min=0, atr_pctile_min=0.0, cooldown=0, enable_shorts=False)
    # After breaking above 5-bar high, should be long at least once
    assert out["signals"].max() == 1
