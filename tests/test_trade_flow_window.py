import json

import pytest

from scripts.collect_trade_flow_window import (
    _checkpoint_after,
    _initialize_window_manifest,
    _recover_cursor,
)


def test_window_manifest_is_frozen_and_timestamped(tmp_path):
    manifest, created = _initialize_window_manifest(
        tmp_path, "btc-eth-window", ["ETHUSDT", "BTCUSDT"], 100_000, 90
    )
    assert created is True
    assert manifest["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert manifest["research_only"] is True
    assert manifest["orders_allowed"] is False
    assert manifest["start_checkpoint_utc"].endswith("Z")
    assert manifest["planned_end_checkpoint_utc"].endswith("Z")

    same, created_again = _initialize_window_manifest(
        tmp_path, "btc-eth-window", ["BTCUSDT", "ETHUSDT"], 100_000, 90
    )
    assert created_again is False
    assert same["start_checkpoint_utc"] == manifest["start_checkpoint_utc"]

    with pytest.raises(ValueError):
        _initialize_window_manifest(
            tmp_path, "btc-eth-window", ["BTCUSDT"], 100_000, 90
        )


def test_cursor_recovery_advances_only_in_segment_order():
    window = {
        "window_id": "test-window",
        "start_checkpoint_utc": "2026-09-01T00:00:00Z",
    }
    first = {
        "segment_id": "segment-000001",
        "continuity_status": "initial",
        "finished_at_utc": "2026-09-01T00:55:00Z",
        "observed_end_exclusive_utc": "2026-09-01T00:55:00Z",
        "normalized_event_count": 100,
        "completed_summary_row_count": 55,
        "gap_seconds_from_previous": 0,
        "overlap_seconds_from_previous": 0,
    }
    cursor = _recover_cursor(None, window, [first])
    assert cursor["next_segment_number"] == 2
    assert cursor["segments_completed"] == 1
    assert cursor["data_through"] == "2026-09-01T00:55:00Z"

    second = dict(first)
    second["segment_id"] = "segment-000003"
    recovered = _recover_cursor(cursor, window, [second])
    assert recovered["next_segment_number"] == 2


def test_checkpoint_records_research_only_boundary():
    previous = {
        "window_id": "test-window",
        "next_segment_number": 1,
        "segments_completed": 0,
        "normalized_event_count": 0,
        "completed_summary_row_count": 0,
        "gap_count": 0,
        "overlap_count": 0,
        "data_through": "2026-09-01T00:00:00Z",
    }
    segment = {
        "segment_id": "segment-000001",
        "continuity_status": "initial",
        "finished_at_utc": "2026-09-01T00:55:00Z",
        "observed_end_exclusive_utc": "2026-09-01T00:55:00Z",
        "normalized_event_count": 100,
        "completed_summary_row_count": 55,
        "gap_seconds_from_previous": 0,
        "overlap_seconds_from_previous": 0,
    }
    updated = _checkpoint_after(previous, segment)
    assert updated["data_through"] == "2026-09-01T00:55:00Z"
    assert updated["last_segment_id"] == "segment-000001"
    assert updated["gap_count"] == 0
    assert updated["overlap_count"] == 0


def test_research_safeguard_fields_are_not_derived_from_trading_runtime():
    source = open("scripts/collect_trade_flow_window.py", encoding="utf-8").read()
    assert "ib_insync" not in source
    assert "placeOrder" not in source
    assert "submit_order" not in source
