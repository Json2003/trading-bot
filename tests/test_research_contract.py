from research.checkpoint import CheckpointError, advance_checkpoint, can_advance

def manifest(cutoff):
    return {"dataset_id": "test", "generated_at": "2026-01-01T00:00:00+00:00", "completed_through": cutoff, "files": [{"path": "x.csv", "sha256": "a" * 64}]}

def test_checkpoint_only_moves_forward():
    previous = {"schema_version": 1, "last_completed_candle": "2026-01-01T00:00:00+00:00", "history": []}
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
    previous = {"schema_version": 1, "last_completed_candle": "2026-01-02T00:00:00+00:00", "history": []}
    assert not can_advance(previous, manifest("2026-01-02T00:00:00+00:00"))
