from datetime import datetime, timedelta, timezone

from research.microstructure_state import StateInput, aggregate_state


NOW = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)


def item(value: float, age_hours: int = 1) -> StateInput:
    observed = NOW - timedelta(hours=age_hours)
    return StateInput(value, observed, observed)


def test_state_is_unknown_until_minimum_fresh_components_exist() -> None:
    result = aggregate_state(NOW, {"funding_persistence": item(0.8), "flow_compression": item(0.6)})
    assert result.status == "unknown"
    assert result.stress_score is None
    assert result.coverage == 0.4


def test_state_uses_fixed_equal_weight_mean() -> None:
    result = aggregate_state(
        NOW,
        {
            "funding_persistence": item(0.8),
            "cross_venue_liquidity": item(0.4),
            "flow_compression": item(0.6),
        },
    )
    assert result.status == "research_only"
    assert result.stress_score == 0.6
    assert result.used_components == (
        "funding_persistence",
        "cross_venue_liquidity",
        "flow_compression",
    )


def test_stale_values_are_not_treated_as_zero() -> None:
    result = aggregate_state(NOW, {"funding_persistence": item(1.0, age_hours=9)})
    assert result.status == "unknown"
    assert result.stress_score is None
    assert "funding_persistence" in result.missing_components


def test_invalid_normalized_values_are_rejected() -> None:
    result = aggregate_state(
        NOW,
        {name: item(0.5) for name in ("funding_persistence", "cross_venue_liquidity", "flow_compression")}
        | {"book_reliability": item(1.1)},
    )
    assert result.status == "research_only"
    assert result.used_components == ("funding_persistence", "cross_venue_liquidity", "flow_compression")
