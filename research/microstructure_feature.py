"""Causal alignment for externally sourced microstructure model candidates.

The Friday brief nominates a measurement; it is never itself a trading signal.
This module accepts measured values with explicit observation and availability
times, and leaves a missing-value flag for gaps in the source.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return result.astimezone(timezone.utc)


def load_measurements(path: Path) -> list[tuple[datetime, datetime, float]]:
    """Read timestamp,available_at,value; reject unavailable or revised history."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"timestamp", "available_at", "value"}.issubset(reader.fieldnames or []):
            raise ValueError("measurements require timestamp,available_at,value")
        rows = []
        for row in reader:
            observed, available = utc(row["timestamp"]), utc(row["available_at"])
            value = float(row["value"])
            if available < observed or not math.isfinite(value):
                raise ValueError("measurement has invalid availability or value")
            rows.append((observed, available, value))
    if len({observed for observed, _, _ in rows}) != len(rows):
        raise ValueError("duplicate observation timestamps need a revision policy")
    return sorted(rows, key=lambda item: item[1])


def align_asof(
    decision_times: list[datetime],
    measurements: list[tuple[datetime, datetime, float]],
    *,
    max_age: timedelta,
) -> list[tuple[float, float]]:
    """Return (value, missing flag), using only values available at decision time.

    A missing value is zero solely as a numeric placeholder; the flag is 1 and
    candidate evaluation must report coverage. Never forward-fill past max_age.
    """
    if max_age <= timedelta(0):
        raise ValueError("max_age must be positive")
    ordered = sorted(measurements, key=lambda item: item[1])
    if decision_times != sorted(decision_times):
        raise ValueError("decision timestamps must be sorted")
    aligned = []
    latest = None
    index = 0
    for decision in decision_times:
        if decision.tzinfo is None:
            raise ValueError("decision timestamps must be timezone-aware")
        while index < len(ordered) and ordered[index][1] <= decision:
            if latest is None or ordered[index][0] > latest[0]:
                latest = ordered[index]
            index += 1
        if latest is None or latest[0] > decision or decision - latest[0] > max_age:
            aligned.append((0.0, 1.0))
        else:
            aligned.append((latest[2], 0.0))
    return aligned
