from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingbot_ibkr.research_context import (
    NewsEvent,
    TradeOutcome,
    TradingMemory,
    align_news_features,
    evaluate_trade_gate,
)


def test_news_alignment_does_not_look_ahead() -> None:
    future = NewsEvent(datetime(2025, 1, 1, 2, tzinfo=timezone.utc), -1.0, 1.0, "macro")
    features = align_news_features(
        [datetime(2025, 1, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, 3, tzinfo=timezone.utc)],
        [future],
    )
    assert features[0].event_count == 0
    assert features[1].event_count == 1
    assert features[1].weighted_sentiment == -1.0


def test_memory_triggers_loss_cooldown() -> None:
    memory = TradingMemory(cooldown_losses=3)
    for index in range(3):
        memory.record(TradeOutcome(datetime(2025, 1, index + 1, tzinfo=timezone.utc), "ema", "range", -1.0, "long"))
    assert memory.snapshot()["cooldown"] is True
    decision = evaluate_trade_gate(1, expected_move_bps=40, expected_cost_bps=10, news=None, memory=memory)
    assert decision.approved is False
    assert decision.reason == "loss_cooldown"


def test_trade_gate_rejects_insufficient_edge_and_high_impact_news() -> None:
    news = align_news_features(
        [datetime(2025, 1, 1, 2, tzinfo=timezone.utc)],
        [NewsEvent(datetime(2025, 1, 1, 1, tzinfo=timezone.utc), 1.0, 0.9, "regulatory")],
    )[0]
    low_edge = evaluate_trade_gate(1, expected_move_bps=12, expected_cost_bps=10, news=news, memory=None)
    assert low_edge.reason == "edge_does_not_cover_costs"
    high_edge = evaluate_trade_gate(1, expected_move_bps=50, expected_cost_bps=10, news=news, memory=None)
    assert high_edge.reason == "high_impact_news_requires_confirmation"


def test_context_rejects_non_finite_values_and_mismatched_series() -> None:
    with pytest.raises(ValueError):
        NewsEvent(datetime(2025, 1, 1, tzinfo=timezone.utc), 0.0, float("nan"))
    with pytest.raises(ValueError):
        from tradingbot_ibkr.research_context import gate_signal_series

        gate_signal_series(
            [1, 0],
            [datetime(2025, 1, 1, tzinfo=timezone.utc)],
            [],
            expected_move_bps=40,
            expected_cost_bps=10,
        )
