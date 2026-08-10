from __future__ import annotations

import pandas as pd

from backtest.optimization.research_loop import _grid_signals_factory


def test_grid_signals_do_not_use_future_prices() -> None:
    builder = _grid_signals_factory(8, 0.05)
    base = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    extended = pd.DataFrame({"close": [100.0, 101.0, 102.0, 1_000.0]})
    assert builder(base)["signals"].tolist() == builder(extended)["signals"].tolist()[:3]
