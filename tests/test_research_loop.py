"""Unit tests for the nightly research loop."""

from __future__ import annotations

import json
import math
from pathlib import Path
import importlib
import os
import sys

# Ensure third-party numpy is used instead of the lightweight repo stub so
# Optuna loads correctly during test discovery.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_site(mod_name: str):
    to_delete = [name for name in sys.modules if name == mod_name or name.startswith(f"{mod_name}.")]
    for name in to_delete:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        try:
            if str(REPO_ROOT) in os.path.abspath(mod_file):
                del sys.modules[name]
        except Exception:
            del sys.modules[name]
    original = sys.path.copy()
    removed_finders: list[object] = []
    for finder in list(sys.meta_path):
        if finder.__class__.__name__ == "_StubFinder":
            sys.meta_path.remove(finder)
            removed_finders.append(finder)
    try:
        repo_paths = {p for p in original if str(REPO_ROOT) in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        module = importlib.import_module(mod_name)
        sys.modules[mod_name] = module
        return module
    finally:
        sys.path = original


_import_site("numpy")
_import_site("pandas")

import numpy as np
import pytest
pytest.importorskip("optuna")
import optuna
import pandas as pd

from backtest.optimization.research_loop import (
    NightlyResearchLoop,
    RegistryEntry,
    ModelRegistry,
    StrategyEvaluation,
    create_non_overlapping_windows,
)


def _make_dataframe(rows: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=rows, freq="H")
    base = np.linspace(100.0, 120.0, rows)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "spot_close": base,
            "futures_close": base * 1.001,
        }
    )
    return df


def test_create_non_overlapping_windows_basic_split() -> None:
    df = _make_dataframe(300)
    windows = create_non_overlapping_windows(df, window_size=120, test_fraction=0.25)
    assert len(windows) == 2
    assert all(len(win.train) > 0 and len(win.test) > 0 for win in windows)
    first = windows[0]
    assert len(first.train) + len(first.test) == 120


def test_evaluate_params_returns_positive_score(tmp_path: Path) -> None:
    df = _make_dataframe(360)
    windows = create_non_overlapping_windows(df, window_size=180, test_fraction=0.3)
    loop = NightlyResearchLoop(
        windows,
        tmp_path / "registry.json",
        trials_range=(1, 1),
        seed=1,
    )
    params = {
        "grid_levels": 10,
        "grid_range_pct": 0.05,
        "momentum_fast": 8,
        "momentum_slow": 21,
        "dca_step_pct": 0.03,
        "dca_max_layers": 3,
        "arb_edge_bps": 8.0,
    }
    score, diagnostics = loop._evaluate_params(windows[0], params)
    assert math.isfinite(score)
    assert diagnostics["oos_sharpe"] >= 0.0


def test_model_registry_persists_top_entries(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path, top_n=2)
    entry = RegistryEntry(
        timestamp="2024-01-01T00:00:00Z",
        window="window_1",
        params={"grid_levels": 12},
        score=1.5,
        train_max_drawdown=-0.05,
        oos_sharpe=1.2,
        oos_max_drawdown=-0.1,
        turnover=0.02,
        diagnostics={"foo": "bar"},
        flag_for_paper_trial=True,
    )
    registry.add(entry)
    assert registry_path.exists()
    data = json.loads(registry_path.read_text())
    assert data[0]["flag_for_paper_trial"] is True


def test_run_updates_registry(tmp_path: Path) -> None:
    df = _make_dataframe(360)
    windows = create_non_overlapping_windows(df, window_size=180, test_fraction=0.3)
    registry_path = tmp_path / "registry.json"
    loop = NightlyResearchLoop(
        windows,
        registry_path,
        trials_range=(1, 1),
        seed=2,
    )
    results = loop.run(max_windows=1)
    assert registry_path.exists()
    saved = json.loads(registry_path.read_text())
    assert 1 <= len(saved) <= 5
    assert all("score" in item for item in saved)
    assert all("train_max_drawdown" in item for item in saved)
    assert len(results) <= 1


def test_drawdown_spike_prunes(monkeypatch, tmp_path: Path) -> None:
    df = _make_dataframe(360)
    windows = create_non_overlapping_windows(df, window_size=180, test_fraction=0.3)
    registry_path = tmp_path / "registry.json"
    loop = NightlyResearchLoop(
        windows,
        registry_path,
        trials_range=(1, 1),
        seed=0,
        drawdown_spike_ratio=1.2,
        drawdown_spike_floor=0.01,
    )

    train_summary = {
        "total_return": 0.05,
        "max_drawdown": -0.01,
        "sharpe": 1.0,
        "sortino": 0.0,
        "profit_factor": 1.0,
        "num_trades": 5,
    }
    test_summary = {
        "total_return": -0.2,
        "max_drawdown": -0.25,
        "sharpe": -0.5,
        "sortino": 0.0,
        "profit_factor": 0.5,
        "num_trades": 5,
    }
    train_returns = [0.01] * 5
    test_returns = [-0.25, 0.0, 0.0, 0.0, 0.0]

    def fake_eval_signal_strategy(self, *_args, **_kwargs):
        return StrategyEvaluation(
            name="signal",
            train_summary=train_summary,
            test_summary=test_summary,
            train_returns=train_returns,
            test_returns=test_returns,
            train_trades=5,
            test_trades=5,
        )

    dca_calls = {"count": 0}

    def fake_simulate_dca(*_args, **_kwargs):
        dca_calls["count"] += 1
        if dca_calls["count"] == 1:
            return train_summary, train_returns, 5
        return test_summary, test_returns, 5

    arb_calls = {"count": 0}

    def fake_simulate_arbitrage(*_args, **_kwargs):
        arb_calls["count"] += 1
        if arb_calls["count"] == 1:
            return train_summary, train_returns, 5
        return test_summary, test_returns, 5

    monkeypatch.setattr(NightlyResearchLoop, "_evaluate_signal_strategy", fake_eval_signal_strategy)
    monkeypatch.setattr("backtest.optimization.research_loop._simulate_dca", fake_simulate_dca)
    monkeypatch.setattr("backtest.optimization.research_loop._simulate_arbitrage", fake_simulate_arbitrage)

    params = {
        "grid_levels": 10,
        "grid_range_pct": 0.05,
        "momentum_fast": 8,
        "momentum_slow": 21,
        "dca_step_pct": 0.03,
        "dca_max_layers": 3,
        "arb_edge_bps": 8.0,
    }

    with pytest.raises(optuna.TrialPruned):
        loop._evaluate_params(windows[0], params)

