from datetime import datetime, timedelta, timezone

import pytest

from scripts.run_cross_exchange_volume_shock import _signal

UTC = timezone.utc


def _series():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    data = {}
    for offset in range(-2, 721):
        timestamp = start + timedelta(hours=offset)
        data[timestamp] = {"close": 100.0, "volume": 10.0}
    signal_timestamp = start + timedelta(hours=720)
    for offset in range(3):
        timestamp = signal_timestamp - timedelta(hours=2 - offset)
        data[timestamp] = {"close": 100.0 + offset * 0.6, "volume": 30.0}
    return data, signal_timestamp


def test_volume_shock_emits_long():
    data, timestamp = _series()
    side, move, current_volume, baseline = _signal(data, timestamp)
    assert side == 1
    assert move == pytest.approx(0.012)
    assert current_volume == pytest.approx(90.0)
    assert baseline == pytest.approx(30.0)


def test_missing_hour_is_not_signal():
    data, timestamp = _series()
    data.pop(timestamp)
    assert _signal(data, timestamp) is None
