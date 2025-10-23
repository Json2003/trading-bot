from __future__ import annotations

import random
from typing import Dict

import numpy as np
import pytest

from tradingbot_ibkr.signal_prediction import SignalPredictor


def _make_features(n: int = 12) -> Dict[str, float]:
    rng = random.Random(42)
    return {f"f{i}": rng.uniform(-1.0, 1.0) for i in range(n)}


@pytest.mark.parametrize("model_type", ["gbm", "lstm", "transformer_orderflow"])
def test_signal_predictor_probability_range(model_type: str) -> None:
    sp = SignalPredictor(model_type=model_type, sequence_length=8)
    # Warm-up for sequence models to ensure there is some history
    for _ in range(10):
        sp.observe(_make_features())
    p = sp.predict(_make_features())
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
