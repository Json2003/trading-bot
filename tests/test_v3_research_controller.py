from __future__ import annotations

import math

import pytest

from scripts.run_momentum_volatility_v3 import (
    CONFIRMATION_HOLDOUT_BARS,
    _confirmation_holdout,
    _folds,
)
from scripts.run_v3_research_controller import (
    SCHEMA_VERSION,
    _read_state,
    _signature,
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
        "promotion_gate": {
            "pass": True,
            "failure_reasons": [],
            "base_walk_forward_median_return_pct": 1.0,
            "stress_walk_forward_median_return_pct": 0.5,
        },
        "full_sample": _result(entries=12, return_pct=full_return),
        "full_sample_stress": _result(entries=12, return_pct=full_return / 2),
        "walk_forward": folds,
        "stress_walk_forward": stress,
        "confirmation_holdout": {
            "base": _result(entries=6, return_pct=1.0),
            "stress": _result(entries=6, return_pct=0.5),
        },
        "horizons": {
            label: _result(entries=1, return_pct=0.25)
            for label in ("1d", "1w", "1m", "1y")
        },
    }


def _report(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "costs": {
            "base": {"fees_bps": 10.0, "slippage_bps": 5.0},
            "stress": {"fees_bps": 20.0, "slippage_bps": 10.0},
        },
        "candidates": {"balanced": candidate},
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
    assert "full sample return is not positive" in result["failure_reasons"]


def test_controller_rejects_nonfinite_results() -> None:
    candidate = _candidate()
    candidate["full_sample"]["return_pct"] = math.nan
    result = evaluate_candidate("balanced", candidate)
    assert result["pass"] is False
    assert any("not finite" in reason for reason in result["failure_reasons"])


def test_controller_requires_confirmation_holdout() -> None:
    candidate = _candidate()
    del candidate["confirmation_holdout"]
    result = evaluate_candidate("balanced", candidate)
    assert result["pass"] is False
    assert "missing confirmation holdout" in result["failure_reasons"]


def test_controller_requires_same_candidate_at_both_sizes() -> None:
    candidate = _candidate()
    reports = {"4000": _report(candidate), "6000": _report(candidate)}
    result = evaluate_readiness(reports)
    assert result["strategy_ready"] is True
    assert result["ready_candidates"] == ["balanced"]


def test_readiness_rejects_missing_required_size() -> None:
    candidate = _candidate()
    result = evaluate_readiness({"4000": _report(candidate)})
    assert result["strategy_ready"] is False
    assert any("exactly $4,000 and $6,000" in reason for reason in result["failure_reasons"])


def test_readiness_rejects_weaker_stress_costs() -> None:
    candidate = _candidate()
    report = _report(candidate)
    report["costs"]["stress"] = {"fees_bps": 5.0, "slippage_bps": 2.0}
    result = evaluate_readiness({"4000": report, "6000": _report(candidate)})
    assert result["strategy_ready"] is False
    assert any("stress costs" in reason for reason in result["failure_reasons"])


def test_validate_data_allows_only_synchronized_bounded_gaps(tmp_path) -> None:
    timestamps = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T01:00:00Z",
        "2023-01-01T03:00:00Z",
        "2023-01-01T04:00:00Z",
    ]
    btc = tmp_path / "BTC.csv"
    eth = tmp_path / "ETH.csv"

    def write_bars(path, values):
        rows = ["timestamp,open,high,low,close,volume"]
        for index, timestamp in enumerate(values):
            price = 100.0 + index
            rows.append(f"{timestamp},{price},{price + 1},{price - 1},{price},1000")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    write_bars(btc, timestamps)
    write_bars(eth, timestamps)
    _, quality = validate_data(btc, eth, minimum_aligned_bars=4)
    assert quality["synchronized_gap_warning"] is True
    assert quality["largest_gap_hours"] == 2.0

    write_bars(eth, [*timestamps[:2], "2023-01-01T02:00:00Z", *timestamps[2:]])
    with pytest.raises(ValueError, match="unsynchronized gaps|identical timestamp coverage"):
        validate_data(btc, eth, minimum_aligned_bars=4)


def test_validate_data_rejects_mismatched_timestamp_coverage(tmp_path) -> None:
    timestamps = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T01:00:00Z",
        "2023-01-01T03:00:00Z",
        "2023-01-01T04:00:00Z",
    ]
    btc = tmp_path / "BTC.csv"
    eth = tmp_path / "ETH.csv"

    def write_bars(path, values):
        rows = ["timestamp,open,high,low,close,volume"]
        for index, timestamp in enumerate(values):
            price = 100.0 + index
            rows.append(f"{timestamp},{price},{price + 1},{price - 1},{price},1000")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    write_bars(btc, timestamps)
    write_bars(eth, [*timestamps, "2023-01-01T05:00:00Z"])
    with pytest.raises(ValueError, match="identical timestamp coverage"):
        validate_data(btc, eth, minimum_aligned_bars=4)


def test_signature_includes_runtime_configuration() -> None:
    data = {"BTC.csv": {"sha256": "a", "bytes": 1}}
    source = {"controller.py": "a"}
    first = _signature(data, source, {"stress_fees_bps": 20.0})
    second = _signature(data, source, {"stress_fees_bps": 100.0})
    assert first != second


def test_confirmation_holdout_is_one_year_and_disjoint() -> None:
    length = 3 * 365 * 24
    folds = _folds(length)
    holdout = _confirmation_holdout(length)
    assert holdout[1] - holdout[0] == CONFIRMATION_HOLDOUT_BARS
    assert folds[-1][1] == holdout[0]


def test_readiness_rejects_stress_component_below_base() -> None:
    candidate = _candidate()
    report = _report(candidate)
    report["costs"]["stress"] = {"fees_bps": 20.0, "slippage_bps": 4.0}
    result = evaluate_readiness({"4000": report, "6000": _report(candidate)})
    assert result["strategy_ready"] is False
    assert any("stress costs" in reason for reason in result["failure_reasons"])


def test_old_or_incomplete_state_cannot_report_ready(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        '{"schema_version": 1, "strategy_ready": true, "iterations_completed": 99}',
        encoding="utf-8",
    )
    state = _read_state(path)
    assert state["schema_version"] == SCHEMA_VERSION
    assert state["strategy_ready"] is False
    path.write_text(
        '{"schema_version": 2, "strategy_ready": true, "iterations_completed": 1, '
        '"consecutive_ready_passes": 3, "history": []}',
        encoding="utf-8",
    )
    state = _read_state(path)
    assert state["strategy_ready"] is False
