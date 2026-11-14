from datetime import datetime, timedelta, timezone

import pytest

from scripts.run_cross_exchange_basis_volume_combination import (
    BASIS_DEVIATION_THRESHOLD,
    _basis_signal,
)

UTC = timezone.utc


def _series(start, values, volume=1.0):
    return {
        start + timedelta(hours=index): {"close": close, "volume": volume}
        for index, close in enumerate(values)
    }


def test_basis_premium_with_volume_shock_emits_long():
    timestamp = datetime(2024, 1, 31, 3, tzinfo=UTC)
    start = timestamp - timedelta(hours=722)
    coinbase = _series(start, [100.0] * 723, volume=1.0)
    binance_close = {key: value["close"] for key, value in coinbase.items()}
    for offset in range(3):
        hour = timestamp - timedelta(hours=2 - offset)
        coinbase[hour] = {"close": 101.0, "volume": 10.0}
        binance_close[hour] = 100.0
    signal = _basis_signal(coinbase, binance_close, timestamp)
    assert signal is not None
    side, deviation, baseline, volume_multiple, _ = signal
    assert side == 1
    assert deviation >= BASIS_DEVIATION_THRESHOLD
    assert baseline == pytest.approx(0.0)
    assert volume_multiple >= 2.0


def test_basis_discount_with_volume_shock_emits_short():
    timestamp = datetime(2024, 1, 31, 3, tzinfo=UTC)
    start = timestamp - timedelta(hours=722)
    coinbase = _series(start, [100.0] * 723, volume=1.0)
    binance_close = {key: value["close"] for key, value in coinbase.items()}
    for offset in range(3):
        hour = timestamp - timedelta(hours=2 - offset)
        coinbase[hour] = {"close": 99.0, "volume": 10.0}
        binance_close[hour] = 100.0
    signal = _basis_signal(coinbase, binance_close, timestamp)
    assert signal is not None
    assert signal[0] == -1


def test_basis_dislocation_without_volume_confirmation_is_not_signal():
    timestamp = datetime(2024, 1, 31, 3, tzinfo=UTC)
    start = timestamp - timedelta(hours=722)
    coinbase = _series(start, [100.0] * 723, volume=1.0)
    binance_close = {key: value["close"] for key, value in coinbase.items()}
    for offset in range(3):
        hour = timestamp - timedelta(hours=2 - offset)
        coinbase[hour] = {"close": 101.0, "volume": 1.0}
        binance_close[hour] = 100.0
    signal = _basis_signal(coinbase, binance_close, timestamp)
    assert signal is not None
    assert signal[0] == 0
