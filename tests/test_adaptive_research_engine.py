import pytest
from research.engine import CandidateScore, classify_result, daily_portfolio_metrics, select_candidate

def test_classification_preserves_cost_sensitive_result():
    assert classify_result(net_return_pct=-1, gross_return_pct=2, drawdown_pct=4, median_block_return_pct=1, trade_count=30, execution_costs=10) == "cost_sensitive"

def test_classification_marks_insufficient_sample():
    assert classify_result(net_return_pct=5, gross_return_pct=6, drawdown_pct=4, median_block_return_pct=1, trade_count=3, execution_costs=1) == "insufficient_sample"

def test_portfolio_drawdown_uses_combined_curve():
    result = daily_portfolio_metrics({"a": [10, -20], "b": [-10, 20]}, {"a": 0.5, "b": 0.5})
    assert result["net_pnl"] == 0
    assert result["max_drawdown_pnl"] == 0

def test_selection_uses_prior_cutoff_and_fixed_tie_break():
    scores = [
        CandidateScore("b", "trend_up", 12, 20, True, "2026-01-01"),
        CandidateScore("a", "trend_up", 12, 20, True, "2026-01-01"),
        CandidateScore("future", "trend_up", 100, 20, True, "2027-01-01"),
    ]
    selected = select_candidate(scores, regime="trend_up", cutoff="2026-12-31")
    assert selected is not None and selected.candidate_id == "a"

def test_selection_stands_aside_without_stress_pass():
    score = CandidateScore("a", "trend_up", 100, 100, False, "2026-01-01")
    assert select_candidate([score], regime="trend_up", cutoff="2026-12-31") is None
