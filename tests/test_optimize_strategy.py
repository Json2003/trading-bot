from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna")

from backtest.io import load_csv
from backtest.optimization import StrategyParams, make_objective, run_trial


def _sample_df():
    return load_csv("backtest/sample_data/sample_ohlcv.csv")


def test_run_trial_produces_metrics():
    df = _sample_df()
    params = StrategyParams(window=20, feature_mix="mom", threshold=0.5)
    score, summary = run_trial(df, params, metric="sharpe", return_summary=True)
    assert isinstance(score, float)
    assert isinstance(summary, dict)
    assert "sharpe" in summary


def test_make_objective_with_fixed_trial():
    df = _sample_df()
    objective = make_objective(df)
    trial = optuna.trial.FixedTrial({"window": 30, "feature_mix": "vol", "thr": 0.4})
    value = objective(trial)
    assert isinstance(value, float)
