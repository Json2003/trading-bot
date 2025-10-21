"""CVaR-aware risk overlay for adjusting model signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class SignalPacket:
    instrument: str
    raw_signal: float
    volatility: float
    predicted_pnl_distribution: Iterable[float]


class CVaRRiskOverlay:
    """Transforms raw model scores into position sizes using CVaR constraints."""

    def __init__(self, confidence: float = 0.95, max_allocation: float = 1.0, capital: float = 1.0) -> None:
        if confidence <= 0 or confidence >= 1:
            raise ValueError("confidence must be in (0, 1)")
        self.confidence = confidence
        self.max_allocation = max_allocation
        self.capital = capital

    def apply(self, signals: Iterable[SignalPacket]) -> Dict[str, float]:
        adjusted: Dict[str, float] = {}
        for packet in signals:
            cvar = self._estimate_cvar(packet.predicted_pnl_distribution)
            if cvar >= 0:
                # downside risk minimal; allow near full allocation scaled by signal
                size = min(self.max_allocation, abs(packet.raw_signal)) * np.sign(packet.raw_signal)
            else:
                risk_budget = min(self.max_allocation, self.capital * 0.02 / (abs(cvar) + 1e-6))
                size = np.clip(packet.raw_signal, -risk_budget, risk_budget)
            adjusted[packet.instrument] = float(size)
        return adjusted

    def _estimate_cvar(self, distribution: Iterable[float]) -> float:
        samples = np.sort(np.array(list(distribution), dtype=float))
        if samples.size == 0:
            return 0.0
        cutoff = int((1 - self.confidence) * samples.size)
        cutoff = max(1, cutoff)
        tail = samples[:cutoff]
        return float(np.mean(tail))
