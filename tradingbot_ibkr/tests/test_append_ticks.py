import pytest
from pathlib import Path

def test_append_ticks_dedup(tmp_path):
    pd = pytest.importorskip("pandas")
    from tradingbot_ibkr.binance_trade_dump_ingest import append_ticks

    out = tmp_path / "BTC_USDT_trades.csv"

    ticks1 = pd.DataFrame(
        [
            {"ts": 1610000000000, "price": 29000.5, "qty": 0.001},
            {"ts": 1610000001000, "price": 29001.0, "qty": 0.002},
        ]
    )
    appended1 = append_ticks(ticks1, out)
    assert appended1 == 2
    df1 = pd.read_csv(out)
    assert len(df1) == 2

    # overlapping second batch: one duplicate and one new row
    ticks2 = pd.DataFrame(
        [
            {"ts": 1610000001000, "price": 29001.0, "qty": 0.002},  # duplicate
            {"ts": 1610000002000, "price": 29010.0, "qty": 0.003},  # new
        ]
    )
    appended2 = append_ticks(ticks2, out)
    # expect only 1 new appended row
    assert appended2 == 1
    df2 = pd.read_csv(out)
    assert len(df2) == 3