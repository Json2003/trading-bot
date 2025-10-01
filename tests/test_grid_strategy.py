import pandas as pd

from backtest.strategies.grid import generate_signals


def test_generate_signals_handles_basic_grid_distribution():
    df = pd.DataFrame({"close": [90, 100, 110]})

    out = generate_signals(df, levels=5, range_pct=0.1)

    assert out["signals"].to_list() == [1, 0, -1]


def test_generate_signals_accepts_capitalised_price_column():
    df = pd.DataFrame({"Close": [100, 99, 101]})

    out = generate_signals(df, levels=4, range_pct=0.02)

    assert set(out["signals"].to_list()).issubset({-1, 0, 1})
