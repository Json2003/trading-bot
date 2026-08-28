from datetime import datetime, timedelta, timezone

from scripts.run_cross_exchange_volume_shock_24h import HOLD_HOURS, _trade


class Bar:
    def __init__(self, timestamp, open_price, close):
        self.timestamp = timestamp
        self.open = open_price
        self.close = close


def test_24_hour_trade_uses_frozen_horizon():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [
        Bar(start + timedelta(hours=hour), 100.0, 100.0)
        for hour in range(26)
    ]
    bars[25].close = 110.0
    result = _trade(bars, 0, 1, "BTC")
    assert HOLD_HOURS == 24
    assert result is not None
    assert result["exit_timestamp"].endswith("01:00:00+00:00")
