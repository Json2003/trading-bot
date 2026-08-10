"""Shared, conservative execution-cost calculations for research diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionCostModel:
    """Round-trip cost assumptions expressed in basis points."""

    spread_bps: float = 12.0
    slippage_bps: float = 8.0
    commission_bps: float = 0.0

    def __post_init__(self) -> None:
        if min(self.spread_bps, self.slippage_bps, self.commission_bps) < 0:
            raise ValueError("execution costs cannot be negative")

    @property
    def per_fill_fraction(self) -> float:
        # Half the quoted spread is paid on each side of a fill.
        return (self.spread_bps / 2.0 + self.slippage_bps + self.commission_bps) / 10_000.0

    @property
    def round_trip_fraction(self) -> float:
        return 2.0 * self.per_fill_fraction

    def net_return(self, gross_return: float, trade_events: int) -> float:
        """Apply a conservative cost to each entry/exit event."""

        if trade_events < 0:
            raise ValueError("trade_events cannot be negative")
        if gross_return <= -1.0:
            return -1.0
        return (1.0 + float(gross_return)) * ((1.0 - self.per_fill_fraction) ** int(trade_events)) - 1.0

