#!/usr/bin/env python3
"""Fail-closed milestone evaluator and capital-growth planner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "multi_asset_rollout.json"


def load_policy(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("mode") != "paper_research_only":
        raise ValueError("rollout policy must remain paper_research_only")
    if policy.get("live_trading_enabled") is not False:
        raise ValueError("live trading must be disabled")
    if policy.get("leverage_enabled") is not False:
        raise ValueError("global leverage must be disabled")
    milestones = policy.get("growth_milestones_usd", [])
    if milestones != sorted(set(milestones)) or not milestones:
        raise ValueError("growth milestones must be unique and ascending")
    return policy


def growth_plan(equity: float, policy: dict[str, Any], *, monthly_contribution: float = 0.0,
                starting_equity: float | None = None, net_contributions: float = 0.0,
                verified_net_pnl: float = 0.0) -> dict[str, Any]:
    if equity < 0 or monthly_contribution < 0:
        raise ValueError("equity and contributions cannot be negative")
    milestones = [float(value) for value in policy["growth_milestones_usd"]]
    reached = [value for value in milestones if equity >= value]
    current = reached[-1] if reached else 0.0
    next_target = next((value for value in milestones if equity < value), None)
    gap = round(max(next_target - equity, 0.0), 2) if next_target else 0.0
    months = math.ceil(gap / monthly_contribution) if next_target and monthly_contribution else None

    reconciliation = None
    if starting_equity is not None:
        reconciled = starting_equity + net_contributions + verified_net_pnl
        reconciliation = {
            "starting_equity_usd": round(starting_equity, 2),
            "net_contributions_usd": round(net_contributions, 2),
            "verified_net_pnl_usd": round(verified_net_pnl, 2),
            "reconciled_equity_usd": round(reconciled, 2),
            "unreconciled_delta_usd": round(equity - reconciled, 2),
        }

    return {
        "settled_equity_usd": round(equity, 2),
        "milestones_usd": milestones,
        "reached_milestones_usd": reached,
        "current_milestone_usd": current,
        "next_milestone_usd": next_target,
        "gap_to_next_milestone_usd": gap,
        "monthly_contribution_usd": round(monthly_contribution, 2),
        "contribution_only_months_to_next": months,
        "return_assumptions_used": False,
        "reconciliation": reconciliation,
    }


def evaluate(equity: float, policy: dict[str, Any], **planner_kwargs: Any) -> dict[str, Any]:
    milestones = [float(value) for value in policy["growth_milestones_usd"]]
    milestone = next((value for value in milestones if equity < value), milestones[-1])
    buffer = float(policy["activation_buffer_usd"])
    milestone_reached = equity >= milestones[0]
    buffer_reached = equity >= milestones[0] + buffer

    sleeves: dict[str, Any] = {}
    for name, config in sorted(policy["sleeves"].items(), key=lambda item: item[1]["priority"]):
        sleeves[name] = {
            "status": "paper_validation_eligible" if milestone_reached else "locked_below_milestone",
            "priority": config["priority"],
            "max_capital_pct": config["max_capital_pct"],
            "live_activation": False,
            "requires_human_approval": True,
            "requirements": list(config["requirements"]),
        }

    return {
        "mode": policy["mode"],
        "equity_usd": round(equity, 2),
        "milestone_usd": milestone,
        "growth_milestones_usd": milestones,
        "buffer_usd": buffer,
        "settled_equity_required": bool(policy["milestone_uses_settled_equity"]),
        "milestone_reached": milestone_reached,
        "buffer_reached": buffer_reached,
        "research_only": True,
        "live_trading_authorized": False,
        "leverage_authorized": False,
        "capital_growth_plan": growth_plan(equity, policy, **planner_kwargs),
        "portfolio_limits": {
            "max_drawdown_pct": policy["max_portfolio_drawdown_pct"],
            "max_open_risk_pct": policy["max_open_risk_pct"],
            "max_sleeve_risk_pct": policy["max_sleeve_risk_pct"],
        },
        "sleeves": sleeves,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity", type=float, required=True)
    parser.add_argument("--monthly-contribution", type=float, default=0.0)
    parser.add_argument("--starting-equity", type=float)
    parser.add_argument("--net-contributions", type=float, default=0.0)
    parser.add_argument("--verified-net-pnl", type=float, default=0.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(
        args.equity, load_policy(args.config),
        monthly_contribution=args.monthly_contribution,
        starting_equity=args.starting_equity,
        net_contributions=args.net_contributions,
        verified_net_pnl=args.verified_net_pnl,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
