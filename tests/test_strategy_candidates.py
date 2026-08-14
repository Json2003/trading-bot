from __future__ import annotations

from pathlib import Path

import pytest

from tradingbot_ibkr.strategy_candidates import (
    CandidateEvidence,
    PaperProbationEvidence,
    StrategyCandidateRegistry,
)


def _research() -> CandidateEvidence:
    return CandidateEvidence(
        source_job_id="job-1",
        source_account_id="final-1",
        dataset_id="sample.csv",
        total_return=0.18,
        max_drawdown=0.08,
        sharpe=1.35,
        profit_factor=1.45,
        trade_count=42,
        score=0.22,
        execution_cost_bps=28.0,
        costs_included=True,
        holdout_passed=True,
        holdout_trade_count=42,
        holdout_total_return=0.11,
        holdout_max_drawdown=0.09,
        holdout_profit_factor=1.30,
    )


def _paper() -> PaperProbationEvidence:
    return PaperProbationEvidence(
        trading_days=25,
        trade_count=80,
        total_return=0.07,
        max_drawdown=0.06,
        profit_factor=1.30,
        reconciliation_passed=True,
        restart_recovery_passed=True,
        duplicate_order_test_passed=True,
        kill_switch_test_passed=True,
    )


def _register(registry: StrategyCandidateRegistry) -> dict[str, object]:
    return registry.register(
        strategy="volume_breakout",
        execution_policy="pullback_limit",
        params={"lookback": 20.0, "volume_multiple": 1.8},
        risk={"risk_per_trade": 0.01, "max_daily_loss_fraction": 0.03},
        research=_research(),
    )


def test_candidate_requires_all_gates_before_export(tmp_path: Path) -> None:
    registry = StrategyCandidateRegistry(tmp_path / "candidates.json")
    candidate = _register(registry)

    with pytest.raises(RuntimeError, match="incomplete gates"):
        registry.export_runtime_config(str(candidate["candidate_id"]))


def test_candidate_can_export_after_probation_and_approval(tmp_path: Path) -> None:
    registry = StrategyCandidateRegistry(tmp_path / "candidates.json")
    candidate = _register(registry)
    candidate_id = str(candidate["candidate_id"])

    registry.record_paper_probation(candidate_id, _paper())
    approved = registry.record_human_approval(
        candidate_id,
        approval_note="Reviewed paper evidence and operational safety gates.",
    )
    exported = registry.export_runtime_config(candidate_id)

    assert approved["live_eligible"] is True
    assert exported["candidate_id"] == candidate_id
    assert exported["strategy"] == "volume_breakout"
    assert exported["export_fingerprint"]


def test_duplicate_candidate_registration_is_idempotent(tmp_path: Path) -> None:
    registry = StrategyCandidateRegistry(tmp_path / "candidates.json")

    first = _register(registry)
    second = _register(registry)

    assert first["candidate_id"] == second["candidate_id"]
    assert len(registry.list()) == 1
