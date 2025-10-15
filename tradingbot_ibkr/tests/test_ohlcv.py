import pytest
from datetime import datetime, timezone

def test_ticks_to_ohlcv_basic():
    pd = pytest.importorskip("pandas")
    from tradingbot_ibkr.binance_trade_dump_ingest import ticks_to_ohlcv

    # two trades in same 1-minute bucket, then another trade in next minute
    ticks = [
        {"ts": 1610000000000, "price": 29000.5, "qty": 0.001},
        {"ts": 1610000001000, "price": 29001.0, "qty": 0.002},
        {"ts": 1610000060000, "price": 29010.0, "qty": 0.003},
    ]

    df = ticks_to_ohlcv(ticks, "1m")
    # expect two bars: first with open=29000.5, high=29001.0, low=29000.5, close=29001.0, volume=0.003
    # second with single tick open=close=29010.0 volume=0.003
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(df) >= 2
    first = df.iloc[0]
    assert float(first["open"]) == pytest.approx(29000.5)
    assert float(first["high"]) == pytest.approx(29001.0)
    assert float(first["low"]) == pytest.approx(29000.5)
    assert float(first["close"]) == pytest.approx(29001.0)
    assert float(first["volume"]) == pytest.approx(0.003)