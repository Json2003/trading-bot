from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.momentum_context import (
    ContextEvent,
    align_context,
    build_context_features,
)


def _ts(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, tzinfo=timezone.utc)


def test_context_is_strictly_as_of_and_does_not_use_future_events() -> None:
    events = [
        ContextEvent(timestamp=_ts(2), sentiment=1.0, impact=0.9),
    ]
    result = align_context([_ts(1), _ts(2), _ts(3)], events)
    assert result["context_sentiment"][0] == 0.0
    assert result["context_risk_event"][0] == 0.0
    assert result["context_sentiment"][1] == pytest.approx(1.0)
    assert result["context_risk_event"][2] == 1.0


def test_context_lookback_expires_old_events() -> None:
    events = [
        ContextEvent(timestamp=_ts(0), sentiment=-1.0, impact=0.8),
    ]
    result = align_context([_ts(0), _ts(25)], events)
    assert result["context_sentiment"][0] == pytest.approx(-1.0)
    assert result["context_event_count"][1] == 0.0


def test_volume_ratio_uses_prior_bars_only() -> None:
    bars = [
        SimpleNamespace(timestamp=_ts(index), volume=100.0)
        for index in range(20)
    ]
    bars.append(SimpleNamespace(timestamp=_ts(20), volume=200.0))
    result = build_context_features(bars)
    assert result["volume_ratio"][19] != result["volume_ratio"][20]
    assert result["volume_ratio"][20] == pytest.approx(2.0)


def test_invalid_event_sentiment_is_rejected() -> None:
    with pytest.raises(ValueError):
        ContextEvent(timestamp=_ts(0), sentiment=1.1, impact=0.2)


def test_non_finite_context_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        ContextEvent(timestamp=_ts(0), sentiment=0.0, impact=float("nan"))
