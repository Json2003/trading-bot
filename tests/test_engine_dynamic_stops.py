import os, sys

# Safe import guard to avoid local pandas.py shadowing site-packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
for shadow in ("pandas", "requests", "ccxt"):
    if shadow in sys.modules:
        del sys.modules[shadow]
# Force site-packages precedence to avoid local pandas.py shadowing
site_paths = [p for p in sys.path if "site-packages" in p]
non_site = [p for p in sys.path if "site-packages" not in p]
sys.path[:] = site_paths + non_site

import pandas as pd
from backtest.engine import ExecConfig, run_backtest


def make_df():
    # synthetic rising market with small pullbacks
    ts = pd.date_range("2024-01-01", periods=60, freq="H", tz="UTC")
    close = pd.Series([100 + i*0.5 + (3 if i % 10 == 0 else 0) for i in range(60)], index=ts)
    high = close + 1
    low = close - 1
    open_ = close
    df = pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1,
    })
    return df


def sig_all_long(df):  # always long regime
    return pd.DataFrame({"signals": 1}, index=df.index)


def test_break_even_and_trail_exit():
    df = make_df()
    cfg = ExecConfig(
        fees_bps=0, slip_bps=0,
        tp_bps=0, sl_bps=0,
        tp_atr_mult=0, sl_atr_mult=10,  # very wide SL so BE/trail triggers first
        atr_period=14,
        notional=1.0,
        risk_per_trade=0.01, max_notional_frac=1.0,
        break_even_atr_mult=0.1, trail_atr_mult=0.1,
        max_bars=0,
    )
    trades, equity, bar_ret = run_backtest(df, sig_all_long, cfg)
    # Expect at least one non-EOD exit due to trailing
    assert len(trades) >= 1
    assert (trades["reason"] == "trail_stop").any() or (trades["reason"] == "tp").any()
    # Ensure pnl computed with notional fraction
    assert "notional_frac" in trades.columns and trades["notional_frac"].gt(0).any()
