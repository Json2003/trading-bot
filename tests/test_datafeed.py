from __future__ import annotations

from datetime import datetime, timedelta, timezone

from engine.datafeed import MarketInstrument, UnifiedDataFeed


class _DummyClient:
    def __init__(self) -> None:
        self._now = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        self._open_interest = 100.0

    def fetch_ticker(self, symbol: str):
        ts = int(self._now.timestamp() * 1000)
        result = {"timestamp": ts, "last": 20000.0, "symbol": symbol}
        self._now += timedelta(minutes=1)
        return result

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 5):
        base = self._now - timedelta(minutes=limit)
        rows = []
        for i in range(limit):
            ts = int((base + timedelta(minutes=i)).timestamp() * 1000)
            price = 20000.0 + i * 10
            rows.append([ts, price, price + 50, price - 50, price + 5, 25.0])
        return rows

    def fetch_funding_rate(self, symbol: str):
        return {"fundingRate": "0.00025"}

    def fetch_open_interest(self, symbol: str):
        self._open_interest += 5.0
        return {"openInterest": self._open_interest}


def test_unified_datafeed_enriches_metrics_with_derivatives_features():
    client = _DummyClient()
    feed = UnifiedDataFeed(
        {"binance": client},
        [MarketInstrument(venue="binance", symbol="BTC/USDT", timeframe="1m")],
        ohlcv_candles=3,
    )

    first = feed.fetch()
    market = first["binance:BTC/USDT"]

    assert market.session == "asia"
    assert market.metrics["session"] == "asia"
    assert abs(market.metrics["funding_rate"] - 0.00025) < 1e-9
    assert abs(market.metrics["open_interest"] - 105.0) < 1e-9
    assert "open_interest_change" not in market.metrics

    second = feed.fetch()
    market2 = second["binance:BTC/USDT"]

    assert market2.session in {"asia", "europe", "us"}
    assert abs(market2.metrics["open_interest"] - 110.0) < 1e-9
    assert abs(market2.metrics["open_interest_change"] - 5.0) < 1e-9
