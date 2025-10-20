from datetime import datetime, timezone

from engine.datafeed import MarketData, ReplayDataFeed


def test_replay_data_feed_cycles_snapshots() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = {
        "binance:BTC/USDT": [
            MarketData(
                venue="binance",
                symbol="BTC/USDT",
                timestamp=timestamp,
                price=1.0,
            ),
            MarketData(
                venue="binance",
                symbol="BTC/USDT",
                timestamp=timestamp,
                price=2.0,
            ),
        ]
    }

    feed = ReplayDataFeed(frame)

    first = feed.fetch()["binance:BTC/USDT"].price
    second = feed.fetch()["binance:BTC/USDT"].price

    assert first == 1.0
    assert second == 2.0

    try:
        feed.fetch()
    except StopIteration:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected StopIteration once snapshots are exhausted")
