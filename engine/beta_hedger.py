"""Beta hedging utilities for portfolio level risk management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class BetaHedgeCfg:
    """Configuration controlling the portfolio beta hedger behaviour."""

    target_beta: float = 0.15
    beta_clip: float = 2.0
    rebalance_thresh: float = 0.05


class BetaHedger:
    """Compute the BTC hedge notional required to reach the target beta."""

    def __init__(self, cfg: BetaHedgeCfg) -> None:
        self.cfg = cfg

    def hedge_notional(
        self,
        exposures_q: Mapping[str, float],
        betas: Mapping[str, float],
        total_equity: float,
    ) -> float:
        if total_equity <= 0:
            return 0.0

        beta_contrib = 0.0
        clip = self.cfg.beta_clip
        for symbol, exposure in exposures_q.items():
            beta = betas.get(symbol, 0.0)
            if beta > clip:
                beta = clip
            elif beta < -clip:
                beta = -clip
            beta_contrib += beta * exposure

        curr_beta = beta_contrib / total_equity
        gap = curr_beta - self.cfg.target_beta
        if abs(gap) < self.cfg.rebalance_thresh:
            return 0.0

        return -gap * total_equity


__all__ = ["BetaHedgeCfg", "BetaHedger"]
