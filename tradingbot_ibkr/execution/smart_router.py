"""Simple smart order router with hooks for game-theoretic responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class VenueStats:
    latency_ms: float
    spread: float
    fill_rate: float
    reaction_score: float = 0.0  # placeholder for fictitious-play adjustments


class SmartOrderRouter:
    """Route orders to venues balancing latency, spread, and behavioural responses."""

    def __init__(self, venues: Dict[str, VenueStats]) -> None:
        if len(venues) < 2:
            raise ValueError("SmartOrderRouter requires at least two venues")
        self.venues = venues

    def route(self, side: str, size: float) -> str:
        scores = {}
        for venue, stats in self.venues.items():
            latency_penalty = stats.latency_ms / 100.0
            spread_penalty = stats.spread
            reaction = stats.reaction_score
            fill_bonus = stats.fill_rate
            score = fill_bonus - (latency_penalty + spread_penalty) - reaction
            scores[venue] = score
        best = max(scores, key=scores.get)
        return best

    def update_stats(self, venue: str, *, latency_ms: Optional[float] = None, spread: Optional[float] = None, fill_rate: Optional[float] = None) -> None:
        stats = self.venues[venue]
        if latency_ms is not None:
            stats.latency_ms = latency_ms
        if spread is not None:
            stats.spread = spread
        if fill_rate is not None:
            stats.fill_rate = fill_rate

    def apply_reaction_model(self, venue: str, opponent_probability: float) -> None:
        """Toy hook approximating fictitious play adjustments."""
        stats = self.venues[venue]
        stats.reaction_score = np.clip(opponent_probability, 0.0, 1.0) * 0.5
