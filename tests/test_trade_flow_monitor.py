import json

from scripts.monitor_binance_trade_flow import FlowAggregator, parse_message


def test_aggregate_trade_infers_aggressor_side_and_notional():
    raw = json.dumps(
        {
            "stream": "btcusdt@aggTrade",
            "data": {
                "e": "aggTrade",
                "s": "BTCUSDT",
                "p": "100.0",
                "q": "2.5",
                "T": 1700000000123,
                "a": 99,
                "m": False,
            },
        }
    )
    events = parse_message(raw)
    assert events == [
        {
            "kind": "aggTrade",
            "event_time_ms": 1700000000123,
            "symbol": "BTCUSDT",
            "price": 100.0,
            "quantity": 2.5,
            "notional": 250.0,
            "aggressor_side": "BUY",
            "aggregate_trade_id": 99,
        }
    ]


def test_liquidation_array_and_book_ticker_are_normalized():
    raw = json.dumps(
        {
            "stream": "!forceOrder@arr",
            "data": [
                {
                    "e": "forceOrder",
                    "E": 1700000001000,
                    "o": {
                        "s": "ETHUSDT",
                        "S": "SELL",
                        "ap": "2000",
                        "z": "3",
                        "T": 1700000001000,
                    },
                }
            ],
        }
    )
    events = parse_message(raw)
    assert events[0]["kind"] == "forceOrder"
    assert events[0]["notional"] == 6000.0
    book = parse_message(
        json.dumps(
            {
                "data": {
                    "e": "bookTicker",
                    "s": "ETHUSDT",
                    "b": "2000",
                    "B": "4",
                    "a": "2001",
                    "A": "2",
                    "T": 1700000001000,
                }
            }
        )
    )[0]
    assert book["best_bid"] == 2000.0
    assert book["ask_quantity"] == 2.0


def test_aggregator_finalizes_only_prior_minutes():
    aggregator = FlowAggregator(["BTCUSDT"], large_trade_notional=100.0)
    first = parse_message(
        json.dumps(
            {
                "data": {
                    "e": "aggTrade",
                    "s": "BTCUSDT",
                    "p": "100",
                    "q": "2",
                    "T": 1700000000123,
                    "a": 1,
                    "m": False,
                }
            }
        )
    )[0]
    assert aggregator.ingest(first) == []
    second = parse_message(
        json.dumps(
            {
                "data": {
                    "e": "aggTrade",
                    "s": "BTCUSDT",
                    "p": "101",
                    "q": "2",
                    "T": 1700000060123,
                    "a": 2,
                    "m": True,
                }
            }
        )
    )[0]
    completed = aggregator.ingest(second)
    assert len(completed) == 1
    assert completed[0]["completed"] is True
    assert completed[0]["buy_notional"] == 200.0
    assert completed[0]["large_buy_notional"] == 200.0
    partial = aggregator.finalize()
    assert partial[0]["completed"] is False
    assert partial[0]["sell_notional"] == 202.0
