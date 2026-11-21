from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import pytest

from tradingbot_ibkr.paper_lab import (
    ExecutionAssumptions,
    StrategyProfile,
    _entry_fill,
)
from tradingbot_ibkr.paper_lab_automation import LabRunSpec, PaperLabAutomationService, _random_profile


def test_entry_cost_excludes_fixed_commission_from_unit_cost() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
            {"timestamp": "2025-01-01T01:00:00Z", "open": 110.0, "high": 111.0, "low": 109.0, "close": 110.0, "volume": 10.0},
        ]
    )
    profile = StrategyProfile(
        account_id="test",
        strategy="ema_momentum",
        execution_policy="next_open_market",
        params={"fast": 5.0, "slow": 15.0},
    )
    assumptions = ExecutionAssumptions(
        spread_bps=12.0,
        slippage_bps=8.0,
        commission_per_order=10.0,
    )
    fill = _entry_fill(frame, 1, profile, 1.0, assumptions)
    assert fill is not None
    entry_price, unit_cost = fill
    expected_price = 110.0 * (1.0 + 14.0 / 10_000.0)
    assert entry_price == pytest.approx(expected_price)
    assert unit_cost == pytest.approx(expected_price - 110.0)


def test_seeded_profile_generation_is_reproducible() -> None:
    first = _random_profile(random.Random(7), 1, 2)
    second = _random_profile(random.Random(7), 1, 2)
    assert first == second


def test_confirmation_evaluates_only_the_locked_finalist(tmp_path: Path) -> None:
    dataset_root = Path(__file__).resolve().parents[1] / "backtest" / "sample_data"
    service = PaperLabAutomationService(dataset_root=dataset_root, artifact_root=tmp_path)
    try:
        job = service.start(LabRunSpec(dataset_id="sample_ohlcv.csv", generations=1, accounts_per_generation=4))
        deadline = time.monotonic() + 10
        while job["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.01)
            job = service.job(job["job_id"])
    finally:
        service.close()
    assert job["state"] == "completed", job.get("error")
    assert len(job["selection_leaderboard"]) >= 2
    assert len(job["final_leaderboard"]) == 1
    assert job["final_leaderboard"][0]["account_id"] == job["locked_account_id"]
    assert job["promotion_allowed"] is False
