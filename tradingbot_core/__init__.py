"""Core utilities shared across trading bot services."""

try:  # pragma: no cover - optional dependency guard
    from .config import ConfigBundle, load_config
except ModuleNotFoundError:  # pragma: no cover - executed when PyYAML missing
    ConfigBundle = None  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]

from .logging_setup import setup_logging
from .backtest_save import save_backtest_results, load_backtest_results
from .backtest_harness import BacktestHarness, BacktestContext, BacktestResult
from .results import deps_fingerprint, save_results
from .monitoring import AlertConfig, MonitoringHub
from .risk import KillSwitch, KillSwitchCfg
from .strategy import Bar, OrderIntent, Strategy
from .momentum import MomentumEMA

try:  # pragma: no cover - optional dependency guard
    from .reconciliation import (
        BacktestEvaluation,
        BacktestProfile,
        BacktestProfileNotFoundError,
        BacktestReconciler,
        MetricEvaluation,
        MetricExpectation,
        load_backtest_profiles,
    )
except ModuleNotFoundError:  # pragma: no cover - executed when yaml missing
    BacktestEvaluation = BacktestProfile = BacktestProfileNotFoundError = None  # type: ignore[assignment]
    BacktestReconciler = MetricEvaluation = MetricExpectation = load_backtest_profiles = None  # type: ignore[assignment]

__all__ = [
    "ConfigBundle",
    "load_config",
    "setup_logging",
    "BacktestHarness",
    "BacktestContext",
    "BacktestResult",
    "AlertConfig",
    "MonitoringHub",
    "KillSwitch",
    "KillSwitchCfg",
    "save_backtest_results",
    "load_backtest_results",
    "deps_fingerprint",
    "save_results",
    "Bar",
    "OrderIntent",
    "Strategy",
    "MomentumEMA",
    "BacktestEvaluation",
    "BacktestProfile",
    "BacktestProfileNotFoundError",
    "BacktestReconciler",
    "MetricEvaluation",
    "MetricExpectation",
    "load_backtest_profiles",
]
