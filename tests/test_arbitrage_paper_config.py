from __future__ import annotations

from pathlib import Path

from scripts.validate_arbitrage_paper_config import validate


def test_conservative_arbitrage_profile_is_paper_only() -> None:
    import yaml

    path = Path("configs/arbitrage-paper.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate(payload) == []


def test_profile_rejects_insufficient_gross_edge() -> None:
    errors = validate(
        {
            "mode": "paper",
            "execution": "synthetic",
            "capital": {"max_trade_notional_usd": 10, "max_total_inventory_usd": 20},
            "costs_bps": {"buy_fee": 10, "sell_fee": 10, "spread": 8, "slippage": 8},
            "gates": {
                "minimum_gross_edge_bps": 20,
                "minimum_net_edge_bps": 15,
                "max_quote_age_seconds": 3,
                "require_both_legs": True,
                "reject_partial_fills": True,
                "reject_transfer_required": True,
            },
            "risk": {"max_consecutive_losses": 3},
        }
    )
    assert "minimum gross edge does not cover modeled costs and net edge" in errors
