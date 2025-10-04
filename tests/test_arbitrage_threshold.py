"""Tests for simple spot-futures arbitrage helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.strategies.arbitrage import check_live_basis, generate_threshold_signals


class DummyExchange:
    """Minimal CCXT-like exchange for deterministic tests."""

    def __init__(self, prices: dict[str, float]):
        self._prices = prices

    def fetch_ticker(self, symbol: str):
        return {"last": self._prices[symbol]}


def test_check_live_basis_generates_positive_signal():
    exchange = DummyExchange({"BTC/USDT": 100.0, "BTC/USDT:USDT": 101.0})
    result = check_live_basis(exchange=exchange, threshold=0.005)
    assert result["signal"] == 1
    assert abs(result["diff"] - 0.01) < 1e-12
    assert result["spot"] == 100.0
    assert result["futures"] == 101.0


def test_check_live_basis_negative_signal():
    exchange = DummyExchange({"BTC/USDT": 100.0, "BTC/USDT:USDT": 99.0})
    result = check_live_basis(exchange=exchange, threshold=0.005)
    assert result["signal"] == -1
    assert abs(result["diff"] + 0.01) < 1e-12


def test_check_live_basis_requires_non_negative_threshold():
    with pytest.raises(ValueError):
        check_live_basis(threshold=-0.1)


def test_generate_threshold_signals_outputs_basis_and_signal():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="1H")
    spot = pd.DataFrame({"timestamp": timestamps, "Close": [100.0, 101.0, 102.0]})
    futures = pd.DataFrame({"timestamp": timestamps, "Close": [101.0, 100.0, 102.5]})

    result = generate_threshold_signals(spot, futures, threshold=0.005)

    expected_basis = [0.01, -0.009900990099009901, 0.004901960784313725]
    for actual, expected in zip(result["basis"], expected_basis):
        assert abs(actual - expected) < 1e-12
    assert result["signal"].to_list() == [1, -1, 0]


def test_generate_threshold_signals_validates_columns():
    spot = pd.DataFrame({"timestamp": [1, 2, 3], "close": [1, 1, 1]})
    futures = pd.DataFrame({"timestamp": [1, 2, 3], "Close": [1, 1, 1]})
    with pytest.raises(ValueError):
        generate_threshold_signals(spot, futures)


def test_generate_threshold_signals_rejects_zero_spot_price():
    timestamps = pd.date_range("2024-01-01", periods=1, freq="1H")
    spot = pd.DataFrame({"timestamp": timestamps, "Close": [0.0]})
    futures = pd.DataFrame({"timestamp": timestamps, "Close": [100.0]})
    with pytest.raises(ValueError):
        generate_threshold_signals(spot, futures)
