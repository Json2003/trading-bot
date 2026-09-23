"""Map legacy controller reports to the governed result schema."""
from __future__ import annotations
from typing import Any

def canonicalize_v3(report: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = report.get("candidates", {})
    if candidate_id not in candidates:
        raise ValueError("candidate not found: " + candidate_id)
    candidate = candidates[candidate_id]
    full = candidate.get("full_sample", {})
    stress = candidate.get("full_sample_stress", {})
    gate = candidate.get("promotion_gate", {})
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "net_return": stress.get("return_pct"),
        "gross_return": full.get("return_pct"),
        "drawdown": stress.get("max_drawdown_pct"),
        "sharpe": None,
        "profit_factor": None,
        "trade_count": stress.get("trades"),
        "execution_costs": None,
        "entries": stress.get("entries"),
        "exits": stress.get("exits"),
        "promotion_gate": gate,
        "metrics_source": "run_momentum_volatility_v3",
        "unavailable_metrics": ["sharpe", "profit_factor", "execution_costs"],
        "research_only": True,
        "orders_placed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
        "risk_limits_changed": False,
        "promotion_allowed": False,
    }
