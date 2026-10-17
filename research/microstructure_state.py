"""Research-only aggregate microstructure state indicator.

This module combines already-measured, point-in-time component values. It does
not fetch data, infer values from report prose, place orders, or alter risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite


@dataclass(frozen=True)
class StateInput:
    """A measured component value and the time it became available."""

    value: float | None
    timestamp: datetime | None
    available_at: datetime | None


@dataclass(frozen=True)
class MicrostructureState:
    status: str
    stress_score: float | None
    coverage: float
    used_components: tuple[str, ...]
    missing_components: tuple[str, ...]


COMPONENTS = (
    "funding_persistence",
    "cross_venue_liquidity",
    "flow_compression",
    "book_reliability",
    "cross_venue_flow",
)


def _valid_at(component: StateInput, decision_at: datetime, max_age: timedelta) -> bool:
    if component.value is None or component.timestamp is None or component.available_at is None:
        return False
    if decision_at.tzinfo is None or component.timestamp.tzinfo is None or component.available_at.tzinfo is None:
        return False
    if not isfinite(component.value) or not 0.0 <= component.value <= 1.0:
        return False
    if component.available_at > decision_at or component.timestamp > decision_at:
        return False
    return decision_at - component.timestamp <= max_age


def aggregate_state(
    decision_at: datetime,
    inputs: dict[str, StateInput],
    *,
    max_age: timedelta = timedelta(hours=8),
    min_components: int = 3,
) -> MicrostructureState:
    """Aggregate normalized stress components using a fixed equal-weight mean.

    The fixed five-component schema and minimum coverage are deliberately
    simple and must be frozen before any validation. This function is
    research-only; callers must not use it as an order or risk gate without a
    separately reviewed promotion decision.
    """
    if decision_at.tzinfo is None:
        raise ValueError("decision_at must include a UTC offset")
    if max_age <= timedelta(0) or not 1 <= min_components <= len(COMPONENTS):
        raise ValueError("invalid freshness or minimum component count")

    valid = tuple(name for name in COMPONENTS if name in inputs and _valid_at(inputs[name], decision_at, max_age))
    missing = tuple(name for name in COMPONENTS if name not in valid)
    coverage = len(valid) / len(COMPONENTS)
    if len(valid) < min_components:
        return MicrostructureState("unknown", None, coverage, valid, missing)
    score = sum(inputs[name].value for name in valid if inputs[name].value is not None) / len(valid)
    return MicrostructureState("research_only", score, coverage, valid, missing)
