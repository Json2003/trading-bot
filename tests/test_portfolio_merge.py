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
from backtest.metrics import summarize


def test_portfolio_merge():
    # pretend two equities offset each other
    ts = pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC")
    eq1 = pd.DataFrame({"timestamp": ts, "equity": [1,1.01,1.00,1.02,1.03]})
    eq2 = pd.DataFrame({"timestamp": ts, "equity": [1,0.99,1.01,1.00,1.02]})
    # simple portfolio avg
    port = eq1.merge(eq2, on="timestamp", suffixes=("_1","_2"))
    port["equity"] = (port["equity_1"] + port["equity_2"]) / 2
    # sanity summarize
    bar_ret = port["equity"].pct_change().fillna(0)
    m = summarize(pd.DataFrame({"pnl":[0.0]}), port[["timestamp","equity"]], bar_ret)
    assert "total_return" in m and m["total_return"] == (port["equity"].iloc[-1]/port["equity"].iloc[0]-1.0)
