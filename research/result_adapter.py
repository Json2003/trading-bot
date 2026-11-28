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


def canonicalize_nyse(report: dict[str, Any], *, arm: str, block: str) -> dict[str, Any]:
    """Map a completed market diagnostic with explicit fraction/USD units.

    Synthetic examples and blocked runs cannot advance a market checkpoint.
    Eligibility for confirmation remains a separate gate in the report.
    """
    from math import isfinite

    from research.engine import research_safety_flags
    from research.nyse_offline import daily_bar_execution_evidence

    if report.get("status") != "historical_diagnostic_complete" or report.get("data_kind") != "market":
        raise ValueError("a completed market diagnostic is required")
    for field, expected in research_safety_flags().items():
        if report.get(field) is not expected:
            raise ValueError(f"unsafe research report flag {field}")
    rows = [r for r in report.get("results", []) if r.get("arm") == arm and r.get("block") == block]
    if len(rows) != 1:
        raise ValueError("select exactly one NYSE result arm and block")
    row = rows[0]
    required_units = {"net_return": "fraction", "drawdown": "negative_fraction", "execution_costs": "USD"}
    for name, unit in required_units.items():
        if row.get("metric_units", {}).get(name) != unit:
            raise ValueError(f"NYSE result unit mismatch for {name}")
    fields = ("net_return", "drawdown", "sharpe", "profit_factor", "trade_count", "execution_costs")
    result = {name: row[name] for name in fields}
    for name, value in result.items():
        if value is not None and (isinstance(value, bool) or not isfinite(float(value))):
            raise ValueError(f"invalid NYSE metric {name}")
    return {"schema_version": 1, "candidate_id": f"{report['experiment_id']}:{arm}:{block}",
            **result, "metric_units": dict(row["metric_units"]),
            "metrics_source": "run_nyse_offline_research", "run_id": report.get("run_id"),
            "observation_ids": list(row["observation_ids"]),
            "minimum_sample_met": row["minimum_sample_met"],
            "eligible_for_confirmation": report["eligible_for_confirmation"],
            "unavailable_metrics": [name for name, value in result.items() if value is None],
            **daily_bar_execution_evidence(), **research_safety_flags()}
