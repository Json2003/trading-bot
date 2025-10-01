import pandas as pd

from backtest.strategies.sma_trend_rsi import generate_signals


def test_generate_signals_emits_long_and_exit():
    closes = [10, 11, 12, 13, 14, 13, 12, 11, 10, 11, 12, 13]
    df = pd.DataFrame({"close": closes})

    out = generate_signals(
        df,
        fast=2,
        slow=3,
        trend_fast=2,
        trend_slow=4,
        rsi_period=3,
        rsi_floor=None,
        rsi_ceiling=None,
    )

    # Expect a long signal once the fast SMA crosses above the slow SMA.
    assert 1 in out["signals"].values

    # Re-enable RSI ceiling to allow exits via overbought conditions.
    out_rsi = generate_signals(
        df,
        fast=2,
        slow=3,
        trend_fast=2,
        trend_slow=4,
        rsi_period=3,
        rsi_floor=None,
        rsi_ceiling=60,
    )

    assert -1 in out_rsi["signals"].values
