from datetime import date, datetime, timezone

import pytest
from scripts.daily_collect import validate_rows


def rows_for(day: date):
    return [
        {
            "timestamp": datetime(
                day.year, day.month, day.day, hour, tzinfo=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "open": "100",
            "high": "101",
            "low": "99",
            "close": "100",
            "volume": "10",
        }
        for hour in range(24)
    ]


def test_daily_collector_accepts_one_complete_utc_day():
    quality = validate_rows(rows_for(date(2026, 9, 8)), date(2026, 9, 8))
    assert quality["rows"] == 24


def test_daily_collector_rejects_missing_hour():
    with pytest.raises(ValueError, match="24 completed"):
        validate_rows(rows_for(date(2026, 9, 8))[:-1], date(2026, 9, 8))


def test_daily_collector_rejects_invalid_ohlcv():
    rows = rows_for(date(2026, 9, 8))
    rows[0]["high"] = "98"
    with pytest.raises(ValueError, match="invalid OHLCV"):
        validate_rows(rows, date(2026, 9, 8))


def test_manifest_without_csv_is_downloaded_and_mergeable(tmp_path, monkeypatch):
    import csv
    import hashlib
    import io
    import zipfile

    from scripts import daily_collect
    from scripts.merge_daily_incremental import merge_symbol

    day = date(2026, 11, 1)
    daily = tmp_path / "daily"
    symbol_dir = daily / "BTCUSDT"
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "2026-11-01.json").write_text("{}")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    for hour in range(24):
        stamp = int(datetime(2026, 11, 1, hour, tzinfo=timezone.utc).timestamp() * 1000)
        writer.writerow([stamp, 100, 101, 99, 100, 10])
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("candles.csv", csv_buffer.getvalue())
    payload = zipped.getvalue()
    digest = hashlib.sha256(payload).hexdigest().encode()
    monkeypatch.setattr(daily_collect, "download", lambda url: digest if url.endswith(".CHECKSUM") else payload)
    result = daily_collect.collect_day("BTCUSDT", day, out_dir=daily, base_url="https://example.invalid")
    assert result["status"] == "downloaded"
    merged = merge_symbol("BTCUSDT", monthly_dir=tmp_path / "monthly", daily_dir=daily, output_dir=tmp_path / "merged")
    assert merged["daily_files_merged"] == 1
    assert merged["rows"] == 24
    assert merged["through"] == "2026-11-01T23:00:00Z"
