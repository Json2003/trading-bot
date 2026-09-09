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
