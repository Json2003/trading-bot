"""Explicit, auditable execution assumptions for research backtests.

This module is intentionally independent from broker adapters.  It turns each
assumption into a per-side basis-point budget and records the assumptions in
the output artifact.  The v3 runner remains research-only; these values never
enable orders or change portfolio risk limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ExecutionModel:
    fee_bps_per_side: float
    spread_bps_per_side: float
    slippage_bps_per_side: float
    impact_bps_per_side: float
    latency_bars: int = 0
    fill_fraction: float = 1.0
    funding_bps_per_bar: float = 0.0
    outage_rejection_rate: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.fee_bps_per_side,
            self.spread_bps_per_side,
            self.slippage_bps_per_side,
            self.impact_bps_per_side,
            self.funding_bps_per_bar,
            self.outage_rejection_rate,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in values):
            raise ValueError("execution costs and rejection rate must be finite and non-negative")
        if self.latency_bars < 0:
            raise ValueError("latency_bars must be non-negative")
        if not 0 < self.fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")
        if self.outage_rejection_rate > 1:
            raise ValueError("outage_rejection_rate must be <= 1")

    @property
    def effective_slippage_bps_per_side(self) -> float:
        # Spread is crossed half on each side; impact is adverse price movement.
        return (
            self.spread_bps_per_side / 2.0
            + self.slippage_bps_per_side
            + self.impact_bps_per_side
        )

    @property
    def effective_fees_bps_per_side(self) -> float:
        return self.fee_bps_per_side

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (
            self.effective_fees_bps_per_side
            + self.effective_slippage_bps_per_side
        )

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "effective_slippage_bps_per_side": self.effective_slippage_bps_per_side,
                "round_trip_bps": self.round_trip_bps,
            }
        )
        return result


BASE_EXECUTION = ExecutionModel(
    fee_bps_per_side=10.0,
    spread_bps_per_side=4.0,
    slippage_bps_per_side=5.0,
    impact_bps_per_side=2.0,
)

STRESS_EXECUTION = ExecutionModel(
    fee_bps_per_side=20.0,
    spread_bps_per_side=10.0,
    slippage_bps_per_side=10.0,
    impact_bps_per_side=8.0,
    latency_bars=1,
    fill_fraction=0.80,
    funding_bps_per_bar=0.5,
    outage_rejection_rate=0.02,
)
