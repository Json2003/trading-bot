from __future__ import annotations

from scripts.run_momentum_volatility_v3 import CONFIRMATION_HOLDOUT_BARS
from scripts.run_v3_exploration import (
    MAX_CANDIDATES,
    candidate_definitions,
    candidate_gate,
    development_folds,
)


def _size_report(*, passed: bool, median: float = 0.25) -> dict[str, object]:
    return {
        "screen_pass": passed,
        "stress_fold_median_return_pct": median,
        "base": {"max_drawdown_pct": 1.0},
        "stress": {"max_drawdown_pct": 1.5},
    }


def test_exploration_suite_is_bounded_and_named() -> None:
    candidates = candidate_definitions()
    assert len(candidates) == MAX_CANDIDATES
    assert {"balanced", "selective", "conservative"} <= set(candidates)


def test_development_folds_do_not_touch_protected_final_year() -> None:
    length = 4 * 365 * 24
    protected_start, folds = development_folds(length)
    assert protected_start == length - CONFIRMATION_HOLDOUT_BARS
    assert len(folds) == 3
    assert folds[-1][1] == protected_start
    assert all(start < end <= protected_start for start, end in folds)


def test_candidate_gate_requires_both_notionals_and_never_promotes() -> None:
    gate = candidate_gate(
        {"4000": _size_report(passed=True), "6000": _size_report(passed=True)}
    )
    assert gate["screen_pass"] is True
    assert gate["eligible_for_promotion"] is False
    assert gate["future_confirmation_required"] is True


def test_candidate_gate_rejects_one_size_failure() -> None:
    gate = candidate_gate(
        {"4000": _size_report(passed=True), "6000": _size_report(passed=False)}
    )
    assert gate["screen_pass"] is False
    assert any("$6000" in reason for reason in gate["failure_reasons"])
