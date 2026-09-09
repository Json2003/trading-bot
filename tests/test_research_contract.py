from research.checkpoint import CheckpointError, advance_checkpoint, can_advance
from scripts.run_governed_research_cycle import iter_results, validate_result


def manifest(cutoff):
    return {
        "dataset_id": "test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "completed_through": cutoff,
        "files": [{"path": "x.csv", "sha256": "a" * 64}],
    }


def valid_result(candidate_id="candidate"):
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "net_return": 1.0,
        "drawdown": 2.0,
        "sharpe": None,
        "profit_factor": None,
        "trade_count": 10,
        "execution_costs": None,
        "research_only": True,
        "orders_placed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
        "risk_limits_changed": False,
        "promotion_allowed": False,
    }


def test_checkpoint_only_moves_forward():
    previous = {
        "schema_version": 1,
        "last_completed_candle": "2026-01-01T00:00:00+00:00",
        "history": [],
    }
    assert can_advance(previous, manifest("2026-01-02T00:00:00+00:00"))
    try:
        advance_checkpoint(previous, manifest("2026-01-01T00:00:00+00:00"))
    except CheckpointError:
        return
    raise AssertionError("backward checkpoint was accepted")


def test_checkpoint_rejects_missing_manifest_fields():
    try:
        can_advance({"schema_version": 1}, {"dataset_id": "x"})
    except CheckpointError:
        return
    raise AssertionError("incomplete manifest was accepted")


def test_checkpoint_does_not_reuse_cutoff():
    previous = {
        "schema_version": 1,
        "last_completed_candle": "2026-01-02T00:00:00+00:00",
        "history": [],
    }
    assert not can_advance(previous, manifest("2026-01-02T00:00:00+00:00"))


def test_nested_candidate_bundle_is_iterable_and_validated():
    payload = {
        "schema_version": 1,
        "candidates": {
            "6000": valid_result("6000"),
            "4000": valid_result("4000"),
        },
    }
    candidates = list(iter_results(payload))
    assert [candidate_id for candidate_id, _ in candidates] == ["4000", "6000"]
    for candidate_id, candidate in candidates:
        validate_result(candidate, candidate_id)


def test_result_contract_rejects_execution_flags():
    result = valid_result()
    result["leverage_enabled"] = True
    try:
        validate_result(result)
    except CheckpointError:
        return
    raise AssertionError("leverage-enabled result was accepted")
