from __future__ import annotations

import pytest

from scripts.run_v3_research_controller import (
    evaluate_candidate,
    evaluate_readiness,
    validate_data,
)


def _result(*, entries: int = 6, return_pct: float = 1.0) -> dict[str, object]:
    return {
        "entries": entries,
        "return_pct": return_pct,
        "pnl": return_pct,
        "kill_events": 0,
        "permanent_halt": False,
    }


def _candidate(*, fold_entries: int = 6, full_return: float = 2.0) -> dict[str, object]:
    folds = [_result(entries=fold_entries, return_pct=value) for value in (-1.0, 2.0, 1.0)]
    stress = [_result(entries=fold_entries, return_pct=value) for value in (-0.5, 1.0, 0.5)]
    return {
        "promotion_gate": {"pass": True, "failure_reasons": []},
        "full_sample": _result(entries=12, return_pct=full_return),
        "walk_forward": folds,
        "stress_walk_forward": stress,
        "horizons": {
            label: _result(entries=1, return_pct=0.25)
            for label in ("1d", "1w", "1m", "1y")
        },
    }


def test_controller_requires_minimum_entries_in_every_fold() -> None:
    candidate = _candidate(fold_entries=4)
    result = evaluate_candidate("balanced", candidate)
    assert result["pass"] is False
    assert any("entries" in reason for reason in result["failure_reasons"])


def test_controller_requires_positive_full_sample_return() -> None:
    candidate = _candidate(full_return=0.0)
    result = evaluate_candidate("balanced", candidate)
    assert result["pass"] is False
    assert "full-sample return is not positive" in result["failure_reasons"]


def test_readiness_requires_same_candidate_at_both_sizes() -> None:
    candidate = _candidate()
    reports = {
        "4000": {"candidates": {"balanced": candidate}},
        "6000": {"candidates": {"balanced": candidate}},
    }
    result = evaluate_readiness(reports)
    assert result["strategy_ready"] is True
    assert result["ready_candidates"] == ["balanced"]


def test_readiness_rejects_candidate_that_passes_only_one_size() -> None:
    candidate = _candidate()
    reports = {
        "4000": {"candidates": {"balanced": candidate}},
        "6000": {"candidates": {"balanced": _candidate(full_return=-1.0)}},
    }
    result = evaluate_readiness(reports)
    assert result["strategy_ready"] is False
    assert result["ready_candidates"] == []


def _write_bars(path, timestamps: list[str]) -> None:
    rows = ["timestamp,open,high,low,close,volume"]
    for index, timestamp in enumerate(timestamps):
        price = 100.0 + index
        rows.append(f"{timestamp},{price},{price + 1},{price - 1},{price},1000")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_validate_data_allows_only_synchronized_bounded_gaps(tmp_path) -> None:
    timestamps = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T01:00:00Z",
        "2023-01-01T03:00:00Z",
        "2023-01-01T04:00:00Z",
    ]
    btc = tmp_path / "BTC.csv"
    eth = tmp_path / "ETH.csv"
    _write_bars(btc, timestamps)
    _write_bars(eth, timestamps)
    _, quality = validate_data(btc, eth, minimum_aligned_bars=4)
    assert quality["synchronized_gap_warning"] is True
    assert quality["largest_gap_hours"] == 2.0

    _write_bars(eth, [*timestamps[:2], "2023-01-01T02:00:00Z", *timestamps[2:]])
    with pytest.raises(ValueError, match="unsynchronized gaps"):
        validate_data(btc, eth, minimum_aligned_bars=4)
