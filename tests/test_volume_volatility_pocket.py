from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_volume_volatility_pocket as mod


def bar(ts, open_price=100.0, high=101.0, low=99.0, close=100.5, volume=1.0):
    return {"timestamp": ts, "open": open_price, "high": high, "low": low, "close": close, "volume": volume}


def test_summary_reports_costs_and_gates():
    result = mod.summary([{"net_pnl": 10.0, "net_return": 10.0 / 3000.0, "execution_cost": 20.0}], "2023-01-01T00:00:00", "2025-01-01T00:00:00")
    assert result["net_pnl"] == 10.0
    assert result["execution_cost"] == 20.0
    assert result["passes_sample_gate"] is False


def test_high_volume_range_signal_generates_trade():
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = [bar(start + timedelta(hours=i), volume=1.0) for i in range(30)]
    rows[20] = bar(start + timedelta(hours=20), high=103.0, low=99.0, close=102.0, volume=3.0)
    rows[22]["open"] = 102.0
    rows[26]["close"] = 105.0
    result = mod.evaluate("BTC", rows, "2023-01-01T00:00:00", "2025-01-01T00:00:00", "2026-08-01T00:00:00")
    assert result["discovery"]["trade_count"] == 1
    assert result["discovery_trade_rows"][0]["side"] == "long"
