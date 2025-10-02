"""Tests for the backtest reconciliation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tradingbot_core.reconciliation import (
    BacktestEvaluation,
    BacktestProfile,
    BacktestProfileNotFoundError,
    BacktestReconciler,
    MetricExpectation,
    load_backtest_profiles,
)


@pytest.fixture(scope="module")
def profiles_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "strategy" / "backtest_profiles.yaml"


def test_load_backtest_profiles(profiles_path: Path) -> None:
    profiles = load_backtest_profiles(profiles_path)

    assert "btc_mean_reversion" in profiles
    btc_profile = profiles["btc_mean_reversion"]
    assert btc_profile.strategy == "Mean Reversion Bands"
    assert {metric.name for metric in btc_profile.metrics} == {
        "win_rate",
        "max_drawdown_pct",
        "sharpe",
    }


def test_evaluate_profile_success(profiles_path: Path) -> None:
    reconciler = BacktestReconciler.from_path(profiles_path)

    evaluation = reconciler.evaluate(
        "btc_mean_reversion",
        {
            "win_rate": 0.58,
            "max_drawdown_pct": 10.5,
            "sharpe": 1.85,
        },
    )

    assert isinstance(evaluation, BacktestEvaluation)
    assert evaluation.passed
    assert evaluation.breaches == ()


def test_evaluate_profile_detects_breach(profiles_path: Path) -> None:
    reconciler = BacktestReconciler.from_path(profiles_path)

    evaluation = reconciler.evaluate(
        "eth_breakout",
        {
            "win_rate": 0.40,
            "max_drawdown_pct": 12.0,
            "profit_factor": 1.10,
        },
    )

    assert not evaluation.passed
    breaches = {metric.name for metric in evaluation.breaches}
    assert breaches == {"win_rate", "profit_factor"}


def test_missing_metric_raises(profiles_path: Path) -> None:
    reconciler = BacktestReconciler.from_path(profiles_path)

    with pytest.raises(KeyError):
        reconciler.evaluate(
            "btc_mean_reversion",
            {
                "win_rate": 0.58,
                "sharpe": 1.75,
            },
        )


def test_get_profile_not_found_raises(profiles_path: Path) -> None:
    reconciler = BacktestReconciler.from_path(profiles_path)

    with pytest.raises(BacktestProfileNotFoundError):
        reconciler.get_profile("unknown")


def test_register_profile_allows_customisation() -> None:
    reconciler = BacktestReconciler()
    custom = BacktestProfile(
        name="custom",
        strategy="Custom",
        market="BTC/USDT",
        timeframe="1h",
        metrics=(
            MetricExpectation(
                name="sharpe",
                target=1.4,
                tolerance=0.1,
                comparison="min",
            ),
        ),
        notes=None,
        tags=("experimental",),
    )

    reconciler.register_profile(custom)
    evaluation = reconciler.evaluate("custom", {"sharpe": 1.45})

    assert isinstance(evaluation, BacktestEvaluation)
    assert evaluation.passed
