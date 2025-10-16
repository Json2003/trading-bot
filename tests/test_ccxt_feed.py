"""Tests for the lightweight CCXT data feed adapter."""

from __future__ import annotations

from typing import Any, Iterable

import pytest

from engine.ccxt_feed import CCXTFeed
from tradingbot_core.strategy import Bar


class DummyExchange:
    def __init__(self, ohlcv_map: dict[str, list[Iterable[Any]]], *, raises: bool = False) -> None:
        self._ohlcv_map = ohlcv_map
        self._raises = raises
        self.calls: list[tuple[str, str, int]] = []

    def fetch_ohlcv(self, symbol: str, *, timeframe: str, limit: int) -> list[Iterable[Any]]:
        self.calls.append((symbol, timeframe, limit))
        if self._raises:
            raise RuntimeError("boom")
        return self._ohlcv_map.get(symbol, [])


def test_latest_bars_returns_exchange_and_any_keys() -> None:
    exchange = DummyExchange({
        "BTC/USDT": [[1, 10, 12, 9, 11, 100]],
    })
    feed = CCXTFeed({"binance": exchange}, ["BTC/USDT"])

    bars = feed.latest_bars()

    assert bars["binance:BTC/USDT"] == Bar(ts=1, open=10, high=12, low=9, close=11, volume=100)
    assert bars["BTC/USDT"] == Bar(ts=1, open=10, high=12, low=9, close=11, volume=100)


def test_latest_bars_prefers_latest_timestamp_for_any_key() -> None:
    exchange_a = DummyExchange({
        "BTC/USDT": [[1, 10, 12, 9, 11, 100]],
    })
    exchange_b = DummyExchange({
        "BTC/USDT": [[2, 20, 22, 19, 21, 200]],
    })
    feed = CCXTFeed({"binance": exchange_a, "kraken": exchange_b}, ["BTC/USDT"])

    bars = feed.latest_bars()

    assert bars["BTC/USDT"].ts == 2
    assert bars["kraken:BTC/USDT"].close == 21


def test_latest_bars_updates_any_key_on_equal_timestamp() -> None:
    exchange_a = DummyExchange({
        "BTC/USDT": [[1, 10, 12, 9, 11, 100]],
    })
    exchange_b = DummyExchange({
        "BTC/USDT": [[1, 20, 22, 19, 21, 200]],
    })
    feed = CCXTFeed({"binance": exchange_a, "kraken": exchange_b}, ["BTC/USDT"])

    bars = feed.latest_bars()

    assert bars["BTC/USDT"].close == 21


def test_latest_bars_skips_failed_requests() -> None:
    ok_exchange = DummyExchange({"ETH/USDT": [[3, 30, 31, 29, 30.5, 300]]})
    failing_exchange = DummyExchange({}, raises=True)
    feed = CCXTFeed({"ok": ok_exchange, "fail": failing_exchange}, ["ETH/USDT"])

    bars = feed.latest_bars()

    # Only the successful exchange should populate entries.
    assert "ok:ETH/USDT" in bars
    assert "fail:ETH/USDT" not in bars
    assert bars["ETH/USDT"].close == 30.5


def test_invalid_ohlcv_payload_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    exchange = DummyExchange({"BTC/USDT": [[1, 2, 3]]})
    feed = CCXTFeed({"binance": exchange}, ["BTC/USDT"])

    with caplog.at_level("WARNING"):
        bars = feed.latest_bars()

    assert bars == {}
    assert "Failed to parse" in "".join(caplog.messages)


def test_requires_exchanges_and_symbols() -> None:
    with pytest.raises(ValueError):
        CCXTFeed({}, ["BTC/USDT"])
    with pytest.raises(ValueError):
        CCXTFeed({"binance": DummyExchange({})}, [])


def test_atr_and_history_are_updated() -> None:
    exchange = DummyExchange({
        "BTC/USDT": [[1, 10, 13, 9, 12, 100]],
    })
    feed = CCXTFeed({"binance": exchange}, ["BTC/USDT"], atr_window=3)

    feed.latest_bars()

    assert feed.atr("BTC/USDT") == 4.0
    assert feed.history("BTC/USDT") == [(1, 10.0, 13.0, 9.0, 12.0, 100.0)]


def test_history_limit_and_missing_symbol() -> None:
    exchange = DummyExchange({"ETH/USDT": [[1, 1, 2, 0.5, 1.5, 10], [2, 1.5, 2.5, 1.0, 2.0, 20]]})
    feed = CCXTFeed({"binance": exchange}, ["ETH/USDT"], atr_window=5)

    feed.latest_bars()

    assert feed.history("UNKNOWN") == []
    assert feed.history("ETH/USDT", limit=1) == [(2, 1.5, 2.5, 1.0, 2.0, 20.0)]
    assert feed.history("ETH/USDT", limit=0) == []


def test_requires_positive_atr_window() -> None:
    with pytest.raises(ValueError):
        CCXTFeed({"binance": DummyExchange({})}, ["BTC/USDT"], atr_window=0)
