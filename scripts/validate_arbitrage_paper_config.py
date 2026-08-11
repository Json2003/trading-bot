"""Validate the conservative paper-only arbitrage profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def validate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("mode") != "paper":
        errors.append("mode must be paper")
    if payload.get("execution") != "synthetic":
        errors.append("execution must be synthetic")

    capital = payload.get("capital", {})
    costs = payload.get("costs_bps", {})
    gates = payload.get("gates", {})
    risk = payload.get("risk", {})

    for key in ("max_trade_notional_usd", "max_total_inventory_usd"):
        if float(capital.get(key, 0)) <= 0:
            errors.append(f"{key} must be positive")
    if float(capital.get("max_trade_notional_usd", 0)) > float(
        capital.get("max_total_inventory_usd", 0)
    ):
        errors.append("max_trade_notional_usd cannot exceed max_total_inventory_usd")

    total_cost = sum(float(costs.get(key, 0)) for key in (
        "buy_fee", "sell_fee", "spread", "slippage"
    ))
    gross = float(gates.get("minimum_gross_edge_bps", 0))
    net = float(gates.get("minimum_net_edge_bps", 0))
    if min(float(value) for value in costs.values()) < 0:
        errors.append("cost assumptions cannot be negative")
    if gross - total_cost < net:
        errors.append("minimum gross edge does not cover modeled costs and net edge")
    if float(gates.get("max_quote_age_seconds", 0)) > 5:
        errors.append("quote age must be at most 5 seconds")
    if risk.get("max_consecutive_losses", 0) < 1:
        errors.append("max_consecutive_losses must be positive")
    if not bool(gates.get("require_both_legs")):
        errors.append("both legs must be required")
    if not bool(gates.get("reject_partial_fills")):
        errors.append("partial fills must be rejected")
    if not bool(gates.get("reject_transfer_required")):
        errors.append("transfer-required opportunities must be rejected")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"valid paper arbitrage profile: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
