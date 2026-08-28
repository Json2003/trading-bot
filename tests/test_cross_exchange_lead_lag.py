from datetime import datetime, timezone

from scripts.run_cross_exchange_lead_lag import _signal

UTC = timezone.utc


def test_positive_coinbase_lead_emits_long():
    timestamp = datetime(2024, 1, 1, 3, tzinfo=UTC)
    coinbase = {
        datetime(2024, 1, 1, 0, tzinfo=UTC): 100.0,
        timestamp: 101.5,
    }
    binance = {
        datetime(2024, 1, 1, 0, tzinfo=UTC): 100.0,
        timestamp: 100.5,
    }
    side, lead_move, execution_move, gap = _signal(coinbase, binance, timestamp)
    assert side == 1
    assert lead_move == 0.015
    assert execution_move == 0.005
    assert gap == 0.01


def test_negative_coinbase_lead_emits_short():
    timestamp = datetime(2024, 1, 1, 3, tzinfo=UTC)
    coinbase = {
        datetime(2024, 1, 1, 0, tzinfo=UTC): 100.0,
        timestamp: 98.0,
    }
    binance = {
        datetime(2024, 1, 1, 0, tzinfo=UTC): 100.0,
        timestamp: 99.5,
    }
    side, *_ = _signal(coinbase, binance, timestamp)
    assert side == -1


def test_incomplete_cross_exchange_window_is_not_signal():
    timestamp = datetime(2024, 1, 1, 3, tzinfo=UTC)
    side = _signal(
        {timestamp: 101.0},
        {timestamp: 100.0},
        timestamp,
    )
    assert side is None
