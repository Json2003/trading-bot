"""Optuna objective helpers for tuning backtest parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict

import optuna

from backtest.engine import ExecConfig, recommended_exec_config, run_backtest
from backtest.metrics import summarize
from backtest.strategies import sma_filtered

FEATURE_CHOICES: tuple[str, ...] = ("mom", "vol", "orderflow")


@dataclass(frozen=True)
class StrategyParams:
    """Container describing the knobs exposed to Optuna."""

    window: int
    feature_mix: str
    threshold: float


def _build_strategy_args(params: StrategyParams) -> Dict[str, Any]:
    """Translate :class:`StrategyParams` into SMA strategy kwargs."""

    window = max(int(params.window), 5)
    feature_mix = params.feature_mix
    if feature_mix not in FEATURE_CHOICES:
        raise ValueError(f"Unsupported feature_mix '{feature_mix}'")

    fast = max(window // 3, 3)
    slow = max(window, fast + 2)
    trend_fast = max(slow, 20)
    trend_slow = max(trend_fast + 40, 60)

    kwargs: Dict[str, Any] = {
        "fast": int(fast),
        "slow": int(slow),
        "trend_fast": int(trend_fast),
        "trend_slow": int(trend_slow),
        "momentum_period": max(int(window // 2), 5),
        "momentum_threshold": 0.0,
        "atr_window": max(int(window * 2), 40),
        "atr_period": max(int(window // 4), 5),
        "atr_pctile": None,
        "rsi_period": max(int(window // 4), 5),
        "rsi_floor": 30.0,
        "rsi_ceiling": 70.0,
        "cooldown": max(int(window // 8), 0),
    }

    threshold = float(params.threshold)

    if feature_mix == "mom":
        kwargs["momentum_threshold"] = threshold * 0.01
        kwargs["atr_pctile"] = None
    elif feature_mix == "vol":
        pct = min(max(threshold, 0.05), 0.95)
        kwargs["atr_pctile"] = pct
        kwargs["momentum_threshold"] = 0.0
    else:  # orderflow
        band = max(5.0, min(25.0, threshold * 20.0))
        center = 50.0
        kwargs["rsi_floor"] = max(5.0, center - band)
        kwargs["rsi_ceiling"] = min(95.0, center + band)
        kwargs["momentum_threshold"] = 0.0

    return kwargs


def run_trial(
    df,
    params: StrategyParams,
    cfg_template: ExecConfig | None = None,
    metric: str = "sharpe",
    *,
    return_summary: bool = False,
):
    """Backtest the strategy with the provided parameters."""

    cfg = replace(cfg_template or recommended_exec_config())

    def _signals(data):
        kwargs = _build_strategy_args(params)
        return sma_filtered.generate_signals(data, **kwargs)

    try:
        trades, equity_curve, bar_returns = run_backtest(df, _signals, cfg)
        summary = summarize(trades, equity_curve, bar_returns)
    except Exception:
        return (float("nan"), {}) if return_summary else float("nan")

    value = summary.get(metric, float("nan"))
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float("nan")

    if return_summary:
        return score, summary
    return score


def make_objective(
    df,
    cfg_template: ExecConfig | None = None,
    metric: str = "sharpe",
) -> Callable[[optuna.trial.Trial], float]:
    """Create an Optuna objective callable bound to the provided dataset."""

    cfg_template = cfg_template or recommended_exec_config()

    def objective(trial: optuna.trial.Trial) -> float:
        params = StrategyParams(
            window=trial.suggest_int("window", 5, 200),
            feature_mix=trial.suggest_categorical("feature_mix", list(FEATURE_CHOICES)),
            threshold=trial.suggest_float("thr", 0.1, 1.0),
        )
        score = run_trial(df, params, cfg_template, metric=metric)
        if not (score == score):  # NaN check
            raise optuna.TrialPruned("Backtest failed; pruning trial.")
        return score

    return objective
