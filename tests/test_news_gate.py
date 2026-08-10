from __future__ import annotations

from datetime import datetime, timezone

from tradingbot_ibkr.research_context import NewsEvent, gate_signal_series


def test_news_gate_blocks_adverse_high_impact_signal() -> None:
    timestamps = [datetime(2025, 1, 1, 1, tzinfo=timezone.utc)]
    events = [NewsEvent(datetime(2025, 1, 1, 0, tzinfo=timezone.utc), -1.0, 0.9, "macro")]
    signals, blocked = gate_signal_series(
        [1],
        timestamps,
        events,
        expected_move_bps=50,
        expected_cost_bps=10,
    )
    assert signals == [0]
    assert blocked["adverse_high_impact_news"] == 1


def test_news_gate_preserves_signal_without_event() -> None:
    signals, blocked = gate_signal_series(
        [-1],
        [datetime(2025, 1, 1, 1, tzinfo=timezone.utc)],
        [],
        expected_move_bps=50,
        expected_cost_bps=10,
    )
    assert signals == [-1]
    assert blocked == {}
