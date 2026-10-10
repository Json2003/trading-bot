from datetime import date, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_synchronized_shock_reversal as mod


def test_summary_is_cost_aware():
    rows = [{"net_pnl": 10.0, "net_return": 10.0 / mod.NOTIONAL, "execution_cost": 20.0}]
    result = mod.summarize(rows, date(2021, 1, 1), date(2021, 2, 1))
    assert result["trade_count"] == 1
    assert result["net_pnl"] == 10.0
    assert result["execution_cost"] == 20.0
    assert result["passes_sample_gate"] is False


def test_tuple_signal_direction_is_used():
    start = date(2021, 1, 1)
    prices = {start + timedelta(days=i): 100.0 for i in range(12)}
    prices[start + timedelta(days=6)] = 90.0
    mod.run_asset._common_signals = {start + timedelta(days=3): (1, -0.06, -0.07)}
    result = mod.run_asset("BTC", prices, start, date(2021, 1, 20), date(2021, 2, 1))
    assert result["discovery"]["trade_count"] == 1
    assert result["discovery_trade_rows"][0]["side"] == "long"
