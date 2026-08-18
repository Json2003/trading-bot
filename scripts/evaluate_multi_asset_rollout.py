#!/usr/bin/env python3
"""Fail-closed account-milestone and multi-asset rollout evaluator.

This policy module is deliberately independent of broker adapters. It can only
authorize research/paper validation; live activation is never returned.
"""
from __future__ import annotations

import argparse
import json
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
    return policy


def evaluate(equity: float, policy: dict[str, Any]) -> dict[str, Any]:
    milestone = float(policy["account_growth_milestone_usd"])
    buffer = float(policy["activation_buffer_usd"])
    settled = bool(policy["milestone_uses_settled_equity"])
    milestone_reached = equity >= milestone
    buffer_reached = equity >= milestone + buffer

    sleeves: dict[str, Any] = {}
    for name, config in sorted(
        policy["sleeves"].items(), key=lambda item: item[1]["priority"]
    ):
        # Crossing the milestone starts research only. No sleeve is promoted here.
        sleeves[name] = {
            "status": "paper_validation_eligible"
            if milestone_reached
            else "locked_below_milestone",
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
        "buffer_usd": buffer,
        "settled_equity_required": settled,
        "milestone_reached": milestone_reached,
        "buffer_reached": buffer_reached,
        "research_only": True,
        "live_trading_authorized": False,
        "leverage_authorized": False,
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(args.equity, load_policy(args.config))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
