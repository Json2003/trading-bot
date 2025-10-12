"""Execution orchestrator wiring strategies, broker and reconciler together."""

from __future__ import annotations

from collections.abc import Mapping
from typing import List, Sequence
import logging

from tradingbot_core.strategy import OrderIntent, Strategy

from tradingbot_ibkr.execution.reconciler import Reconciler, RiskLimits


def _lookup(cfg: object, key: str, *aliases: str) -> float:
    """Return ``key`` from ``cfg`` supporting attribute or mapping access."""

    candidates: Sequence[str] = (key, *aliases)
    for name in candidates:
        if isinstance(cfg, Mapping) and name in cfg:
            return float(cfg[name])  # type: ignore[index]
        if hasattr(cfg, name):
            return float(getattr(cfg, name))
    alias_text = " or ".join(candidates)
    raise AttributeError(f"risk configuration missing '{alias_text}'")


class Orchestrator:
    def __init__(
        self,
        strategies: List[Strategy],
        broker,
        risk_cfg,
        portfolio_book,
        datafeed,
        logger: logging.Logger | None = None,
    ) -> None:
        self.strategies = strategies
        self.broker = broker
        self.portfolio = portfolio_book
        self.datafeed = datafeed
        self.log = logger or logging.getLogger("orchestrator")

        limits = RiskLimits(
            max_daily_loss_pct=_lookup(risk_cfg, "max_daily_loss_pct"),
            kill_switch_drawdown_pct=_lookup(risk_cfg, "kill_switch_drawdown_pct"),
            max_position_risk_pct=_lookup(risk_cfg, "max_position_risk_pct", "per_trade_risk_pct"),
        )
        self.reconciler = Reconciler(broker, limits=limits, logger=self.log)

    def step(self) -> None:
        bars = self.datafeed.latest_bars()
        intents: List[OrderIntent] = []
        for strategy in self.strategies:
            try:
                intents.extend(strategy.on_bar(bars))
            except Exception as exc:  # pragma: no cover - defensive logging
                self.log.exception("%s on_bar error: %s", strategy.name, exc)

        for intent in intents:
            order = self.broker.intent_to_order(intent)
            self.reconciler.submit_idempotent(order)

        if self.reconciler.check_kill_switch(self.portfolio.equity_curve):
            self.log.error("Kill-switch hit; stopping.")
            raise SystemExit(2)


__all__ = ["Orchestrator"]
