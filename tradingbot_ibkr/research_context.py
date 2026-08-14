"""News-aware and trading-memory context for paper research decisions.

This module deliberately separates context collection from signal generation.
It never fabricates news, predicts sentiment, or enables live trading by
itself. All news features are strictly as-of timestamped to prevent leakage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable, Mapping


def _utc(value: Any) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class NewsEvent:
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


@dataclass(frozen=True, slots=True)
class NewsFeatures:
    timestamp: datetime
    event_count: int
    weighted_sentiment: float
    max_impact: float
    risk_event: bool
    latest_category: str | None


def align_news_features(
    bar_timestamps: Iterable[Any],
    events: Iterable[NewsEvent],
    *,
    lookback: timedelta = timedelta(hours=24),
    risk_impact_threshold: float = 0.75,
) -> list[NewsFeatures]:
    """Build strictly as-of news features for each market-bar timestamp."""

    if lookback.total_seconds() <= 0:
        raise ValueError("lookback must be positive")
    if not math.isfinite(risk_impact_threshold) or risk_impact_threshold < 0:
        raise ValueError("risk_impact_threshold must be finite and non-negative")
    ordered_events = sorted(list(events), key=lambda event: _utc(event.timestamp))
    output: list[NewsFeatures] = []
    for raw_timestamp in bar_timestamps:
        timestamp = _utc(raw_timestamp)
        recent = [
            event
            for event in ordered_events
            if timestamp - lookback <= _utc(event.timestamp) <= timestamp
        ]
        weight = sum(event.impact for event in recent)
        sentiment = (
            sum(event.sentiment * event.impact for event in recent) / weight
            if weight > 0
            else 0.0
        )
        latest = max(recent, key=lambda event: _utc(event.timestamp), default=None)
        output.append(
            NewsFeatures(
                timestamp=timestamp,
                event_count=len(recent),
                weighted_sentiment=float(sentiment),
                max_impact=max((event.impact for event in recent), default=0.0),
                risk_event=any(event.impact >= risk_impact_threshold for event in recent),
                latest_category=latest.category if latest else None,
            )
        )
    return output


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    timestamp: datetime
    strategy: str
    regime: str
    pnl: float
    signal_type: str


class TradingMemory:
    """Bounded, serializable memory of realized trade outcomes."""

    def __init__(self, *, max_records: int = 500, cooldown_losses: int = 3) -> None:
        if max_records < 1 or cooldown_losses < 1:
            raise ValueError("memory limits must be positive")
        self._records: deque[TradeOutcome] = deque(maxlen=max_records)
        self._cooldown_losses = cooldown_losses

    def record(self, outcome: TradeOutcome) -> None:
        if not math.isfinite(float(outcome.pnl)):
            raise ValueError("trade outcome pnl must be finite")
        self._records.append(outcome)

    def snapshot(self, *, strategy: str | None = None) -> dict[str, Any]:
        records = [item for item in self._records if strategy is None or item.strategy == strategy]
        wins = sum(item.pnl > 0 for item in records)
        losses = sum(item.pnl < 0 for item in records)
        consecutive_losses = 0
        for item in reversed(records):
            if item.pnl < 0:
                consecutive_losses += 1
            else:
                break
        return {
            "trade_count": len(records),
            "win_rate": wins / len(records) if records else 0.0,
            "expectancy": sum(item.pnl for item in records) / len(records) if records else 0.0,
            "wins": wins,
            "losses": losses,
            "consecutive_losses": consecutive_losses,
            "cooldown": consecutive_losses >= self._cooldown_losses,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "cooldown_losses": self._cooldown_losses,
            "records": [
                {
                    **asdict(item),
                    "timestamp": _utc(item.timestamp).isoformat(),
                }
                for item in self._records
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradingMemory":
        memory = cls(cooldown_losses=int(payload.get("cooldown_losses", 3)))
        for item in payload.get("records", []):
            memory.record(
                TradeOutcome(
                    timestamp=_utc(item["timestamp"]),
                    strategy=str(item["strategy"]),
                    regime=str(item["regime"]),
                    pnl=float(item["pnl"]),
                    signal_type=str(item["signal_type"]),
                )
            )
        return memory


@dataclass(frozen=True, slots=True)
class GateDecision:
    approved: bool
    reason: str
    adjusted_signal: int


def evaluate_trade_gate(
    raw_signal: int,
    *,
    expected_move_bps: float,
    expected_cost_bps: float,
    news: NewsFeatures | None,
    memory: TradingMemory | None,
    minimum_edge_bps: float = 5.0,
) -> GateDecision:
    """Apply cost, news-risk, and recent-loss gates to a candidate signal."""

    signal = 1 if raw_signal > 0 else -1 if raw_signal < 0 else 0
    if signal == 0:
        return GateDecision(False, "no_signal", 0)
    numeric_inputs = (expected_move_bps, expected_cost_bps, minimum_edge_bps)
    if (
        not all(math.isfinite(float(value)) for value in numeric_inputs)
        or expected_move_bps < 0
        or expected_cost_bps < 0
        or minimum_edge_bps < 0
    ):
        return GateDecision(False, "invalid_cost_inputs", 0)
    if expected_move_bps - expected_cost_bps < minimum_edge_bps:
        return GateDecision(False, "edge_does_not_cover_costs", 0)
    if memory is not None and memory.snapshot().get("cooldown"):
        return GateDecision(False, "loss_cooldown", 0)
    if news is not None and news.risk_event:
        if news.weighted_sentiment * signal < 0:
            return GateDecision(False, "adverse_high_impact_news", 0)
        return GateDecision(False, "high_impact_news_requires_confirmation", 0)
    return GateDecision(True, "approved", signal)


def gate_signal_series(
    signals: Iterable[int],
    timestamps: Iterable[Any],
    events: Iterable[NewsEvent],
    *,
    expected_move_bps: float,
    expected_cost_bps: float,
    minimum_edge_bps: float = 5.0,
) -> tuple[list[int], dict[str, int]]:
    """Apply the news gate to a signal series and return block diagnostics."""

    signal_values = list(signals)
    timestamp_values = list(timestamps)
    event_values = list(events)
    if len(signal_values) != len(timestamp_values):
        raise ValueError("signals and timestamps must have equal length")
    features = align_news_features(timestamp_values, event_values)
    gated: list[int] = []
    blocked: dict[str, int] = {}
    for raw_signal, news in zip(signal_values, features):
        decision = evaluate_trade_gate(
            int(raw_signal),
            expected_move_bps=expected_move_bps,
            expected_cost_bps=expected_cost_bps,
            news=news,
            memory=None,
            minimum_edge_bps=minimum_edge_bps,
        )
        if int(raw_signal) != 0 and not decision.approved:
            blocked[decision.reason] = blocked.get(decision.reason, 0) + 1
        gated.append(decision.adjusted_signal)
    return gated, blocked


__all__ = [
    "GateDecision",
    "NewsEvent",
    "NewsFeatures",
    "TradeOutcome",
    "TradingMemory",
    "align_news_features",
    "evaluate_trade_gate",
    "gate_signal_series",
]
