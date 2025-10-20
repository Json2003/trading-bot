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
import pandas as pd

from backtest.optimization.research_loop import (
    NightlyResearchLoop,
    RegistryEntry,
    ModelRegistry,
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
    assert len(results) <= 1

