"""Causal sentiment, event-risk, and liquidity features for research.

This module is intentionally offline and research-only.  It accepts a timestamped
CSV supplied by the operator; it never fetches news, places orders, or changes
paper/live configuration.  Every event is used only at or after its timestamp.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ContextEvent:
    timestamp: datetime
    sentiment: float
    impact: float
    category: str = "unknown"
    source: str = "unknown"

    def __post_init__(self) -> None:
        timestamp = _utc(self.timestamp)
        sentiment = float(self.sentiment)
        impact = float(self.impact)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "sentiment", sentiment)
        object.__setattr__(self, "impact", impact)
        if not math.isfinite(sentiment) or not -1.0 <= sentiment <= 1.0:
            raise ValueError("sentiment must be finite and between -1 and 1")
        if not math.isfinite(impact) or impact < 0:
            raise ValueError("impact must be finite and non-negative")


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def load_context_events(path: Path) -> list[ContextEvent]:
    """Load an operator-supplied timestamped context CSV.

    Required columns are timestamp,sentiment,impact.  Events are sorted and
    validated before use.  This function does not infer sentiment from future
    prices and does not permit malformed timestamps.
    """

    events: list[ContextEvent] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "sentiment", "impact"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"context CSV missing columns: {sorted(missing)}")
        for row in reader:
            events.append(
                ContextEvent(
                    timestamp=_utc(row["timestamp"]),
                    sentiment=float(row["sentiment"]),
                    impact=float(row["impact"]),
                    category=str(row.get("category", "unknown")),
                    source=str(row.get("source", "unknown")),
                )
            )
    return sorted(events, key=lambda event: event.timestamp)


def _prior_median(values: Sequence[float], index: int, window: int) -> float:
    prior = [float(value) for value in values[max(0, index - window):index] if float(value) > 0]
    return median(prior) if prior else math.nan


def _volume_ratios(volumes: Sequence[float], window: int) -> list[float]:
    ratios: list[float] = []
    for index, volume in enumerate(volumes):
        baseline = _prior_median(volumes, index, window)
        ratios.append(float(volume) / baseline if baseline > 0 else math.nan)
    return ratios


def align_context(
    timestamps: Iterable[Any],
    events: Sequence[ContextEvent],
    *,
    lookback: timedelta = timedelta(hours=24),
    risk_impact_threshold: float = 0.75,
) -> dict[str, list[float]]:
    """Align events strictly as-of each bar timestamp."""

    if lookback.total_seconds() <= 0:
        raise ValueError("lookback must be positive")
    if not math.isfinite(risk_impact_threshold) or risk_impact_threshold < 0:
        raise ValueError("risk_impact_threshold must be finite and non-negative")
    ordered = sorted(list(events), key=lambda event: event.timestamp)
    output = {
        "context_sentiment": [],
        "context_impact": [],
        "context_event_count": [],
        "context_risk_event": [],
    }
    for raw_timestamp in timestamps:
        timestamp = _utc(raw_timestamp)
        recent = [
            event
            for event in ordered
            if timestamp - lookback <= event.timestamp <= timestamp
        ]
        total_impact = sum(event.impact for event in recent)
        sentiment = (
            sum(event.sentiment * event.impact for event in recent) / total_impact
            if total_impact > 0
            else 0.0
        )
        output["context_sentiment"].append(float(sentiment))
        output["context_impact"].append(float(max((event.impact for event in recent), default=0.0)))
        output["context_event_count"].append(float(len(recent)))
        output["context_risk_event"].append(
            1.0 if any(event.impact >= risk_impact_threshold for event in recent) else 0.0
        )
    return output


def build_context_features(
    bars: Sequence[Any],
    events: Sequence[ContextEvent] | None = None,
    *,
    volume_window: int = 20,
    lookback_hours: int = 24,
) -> dict[str, list[float]]:
    """Build causal context and liquidity proxies for OHLCV bars."""

    if volume_window <= 0 or lookback_hours <= 0:
        raise ValueError("volume_window and lookback_hours must be positive")
    timestamps = [bar.timestamp for bar in bars]
    volumes = [float(bar.volume) for bar in bars]
    context = align_context(
        timestamps,
        list(events or []),
        lookback=timedelta(hours=lookback_hours),
    )
    context["volume_ratio"] = _volume_ratios(volumes, volume_window)
    return context


__all__ = [
    "ContextEvent",
    "align_context",
    "build_context_features",
    "load_context_events",
]
