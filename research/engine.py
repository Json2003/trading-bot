"""Pure research utilities for classification and adaptive selection."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    regime: str
    expectancy_bps: float
    observations: int
    stress_pass: bool
    prior_cutoff: str


def classify_result(
    *,
    net_return_pct: float,
    gross_return_pct: float,
    drawdown_pct: float,
    median_block_return_pct: float | None,
    trade_count: int,
    execution_costs: float,
    required_trades: int = 20,
) -> str:
    """Classify a result without treating every failed gate as rejection."""
    values = (net_return_pct, gross_return_pct, drawdown_pct, execution_costs)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("result metrics must be finite")
    if trade_count < required_trades:
        return "insufficient_sample"
    if gross_return_pct <= 0:
        return "negative_gross_edge"
    if net_return_pct <= 0:
        return "cost_sensitive"
    if median_block_return_pct is not None and median_block_return_pct <= 0:
        return "unstable_regime"
    if drawdown_pct > 50:
        return "unstable_regime"
    return "promising_needs_adjustment"


def daily_portfolio_metrics(
    daily_pnl: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Combine aligned daily sleeve P&L; never average sleeve drawdowns."""
    if not daily_pnl or set(daily_pnl) != set(weights):
        raise ValueError("daily P&L and weights must contain the same candidates")
    lengths = {len(values) for values in daily_pnl.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all aligned daily P&L series must share a non-zero length")
    weight_total = sum(weights.values())
    if abs(weight_total - 1.0) > 1e-9:
        raise ValueError("portfolio weights must sum to one")
    combined = [
        sum(float(daily_pnl[name][index]) * weights[name] for name in daily_pnl)
        for index in range(next(iter(lengths)))
    ]
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in combined:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    deviation = pstdev(combined)
    return {
        "net_pnl": sum(combined),
        "daily_pnl": combined,
        "max_drawdown_pnl": drawdown,
        "sharpe_proxy": mean(combined) / deviation if deviation else None,
        "observations": len(combined),
        "weights": dict(weights),
    }


def select_candidate(
    scores: Sequence[CandidateScore],
    *,
    regime: str,
    minimum_observations: int = 20,
    minimum_expectancy_bps: float = 5.0,
    cutoff: str,
) -> CandidateScore | None:
    """Select using prior data only, with a deterministic tie break."""
    eligible = [
        item for item in scores
        if item.regime == regime
        and item.observations >= minimum_observations
        and item.expectancy_bps >= minimum_expectancy_bps
        and item.stress_pass
        and item.prior_cutoff <= cutoff
    ]
    return sorted(eligible, key=lambda item: (-item.expectancy_bps, item.candidate_id))[0] if eligible else None


def research_safety_flags() -> dict[str, bool]:
    return {
        "research_only": True,
        "orders_placed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
        "risk_limits_changed": False,
        "promotion_allowed": False,
    }
