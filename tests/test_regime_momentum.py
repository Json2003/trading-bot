from __future__ import annotations

import pandas as pd
import pytest

from backtest.strategies.regime_momentum import generate_signals


def test_regime_momentum_emits_aligned_signal_column() -> None:
    close = [100 + i * 0.1 for i in range(260)]
    frame = pd.DataFrame({"close": close, "high": [x + 1 for x in close], "low": [x - 1 for x in close]})
    signals = generate_signals(frame, fast=5, slow=10, regime=30)
    assert len(signals) == len(frame)
    assert set(signals["signals"].unique()).issubset({-1, 0, 1})


def test_regime_momentum_rejects_invalid_order() -> None:
    frame = pd.DataFrame({"close": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20})
    with pytest.raises(ValueError):
        generate_signals(frame, fast=20, slow=10, regime=30)
