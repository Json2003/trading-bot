import pandas as pd
import pytest

from backtest.strategies.sma_trend import generate_signals


def test_generate_signals_emits_buy_and_sell_signals():
    df = pd.DataFrame(
        {
            "close": [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
            ]
        }
    )

    out = generate_signals(df, fast=2, slow=3, trend_fast=2, trend_slow=4)

    assert 1 in list(out["signal"].values)
    assert -1 in list(out["signal"].values)


def test_generate_signals_does_not_mutate_input():
    df = pd.DataFrame({"close": [10, 11, 12, 13]})
    original_cols = list(df.columns)

    generate_signals(df)

    assert list(df.columns) == original_cols


def test_generate_signals_requires_close_column():
    df = pd.DataFrame({"price": [1, 2, 3]})

    with pytest.raises(ValueError):
        generate_signals(df)
