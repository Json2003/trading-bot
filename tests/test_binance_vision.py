from __future__ import annotations

import zipfile

import pandas as pd

from scripts.fetch_binance_klines import (
    _read_member,
    archive_url,
    read_kline_archive,
)


def test_archive_urls_use_official_spot_layout() -> None:
    assert archive_url(
        "spot", "BTCUSDT", "1h", pd.Timestamp("2025-01-01").date(), monthly=True
    ) == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
        "BTCUSDT-1h-2025-01.zip"
    )
    assert "/daily/klines/BTCUSDT/1h/2025/01/02/" in archive_url(
        "spot", "BTCUSDT", "1h", pd.Timestamp("2025-01-02").date(), monthly=False
    )


def test_kline_reader_normalizes_microsecond_timestamps() -> None:
    row = "1735689600000000,100,101,99,100.5,12,1735693199999999,1200,1,6,600,0\n"
    frame = _read_member(row.encode(), "BTCUSDT-1h.csv")
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_kline_archive_reader_accepts_header_member(tmp_path) -> None:
    archive_path = tmp_path / "BTCUSDT-1h-2025-01.zip"
    contents = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,number_of_trades,"
        "taker_buy_base_volume,taker_buy_quote_volume,ignore\n"
        "1735689600000,100,101,99,100.5,12,1735693199999,1200,1,6,600,0\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("BTCUSDT-1h-2025-01.csv", contents)
    frame = read_kline_archive(archive_path)
    assert len(frame) == 1
    assert frame.loc[0, "close"] == 100.5
