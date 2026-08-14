from __future__ import annotations

import pytest

from backtest.optimization.execution_model import ExecutionCostModel


def test_cost_model_reduces_positive_return() -> None:
    model = ExecutionCostModel(spread_bps=12, slippage_bps=8)
    assert model.per_fill_fraction == pytest.approx(0.0014)
    assert model.net_return(0.10, 10) < 0.10


def test_cost_model_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        ExecutionCostModel(slippage_bps=-1)
    with pytest.raises(ValueError):
        ExecutionCostModel().net_return(0.1, -1)


def test_cost_model_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError):
        ExecutionCostModel(slippage_bps=float("nan"))
