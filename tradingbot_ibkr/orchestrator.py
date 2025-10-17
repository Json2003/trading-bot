"""Execution orchestrator wiring strategies, broker and reconciler together."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Dict, List
import logging

from tradingbot_core.risk import KillSwitch, KillSwitchCfg
from tradingbot_core.strategy import Bar, OrderIntent, Strategy

from tradingbot_ibkr.execution.reconciler import Reconciler, RiskLimits
from tradingbot_ibkr.money_engine import qty_from_risk


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


def _resolve_equity_curve(portfolio: object) -> list[float]:
    """Extract an equity curve snapshot from ``portfolio`` if available."""

    curve: list[float] = []
    if hasattr(portfolio, "equity_curve"):
        try:
            data = getattr(portfolio, "equity_curve")
            if isinstance(data, Sequence):
                curve = [float(value) for value in data]
        except Exception:  # pragma: no cover - defensive guard
            curve = []
    return curve


class Orchestrator:
    def __init__(
        self,
        strategies: List[Strategy],
        broker,
        risk_cfg,
        portfolio_book,
        datafeed,
        logger: logging.Logger | None = None,
        atr_mult: float = 2.0,
    ) -> None:
        self.strategies = strategies
        self.broker = broker
        self.portfolio = portfolio_book
        self.datafeed = datafeed
        self.atr_mult = float(atr_mult)
        self.log = logger or logging.getLogger("orchestrator")

        max_daily_loss = _lookup(risk_cfg, "max_daily_loss_pct")
        kill_drawdown = _lookup(risk_cfg, "kill_switch_drawdown_pct")
        max_position_risk = _lookup(risk_cfg, "max_position_risk_pct", "per_trade_risk_pct")

        limits = RiskLimits(
            max_daily_loss_pct=max_daily_loss,
            kill_switch_drawdown_pct=kill_drawdown,
            max_position_risk_pct=max_position_risk,
        )
        self.limits = limits
        self.reconciler = Reconciler(broker, limits=limits, logger=self.log)

        curve = _resolve_equity_curve(self.portfolio)
        if curve:
            start_equity = curve[-1]
        else:
            start_equity = float(getattr(self.portfolio, "total_equity", 0.0))
        self._equity_curve: list[float] = curve if curve else [float(start_equity)]

        kill_cfg = KillSwitchCfg(max_dd_pct=kill_drawdown, max_daily_loss_pct=max_daily_loss)
        self.kill = KillSwitch(kill_cfg, start_equity=float(start_equity))

    def _portfolio_equity(self) -> float:
        """Return aggregate portfolio equity using available attributes."""

        equity_map = getattr(self.portfolio, "strategy_equity", None)
        if isinstance(equity_map, Mapping) and equity_map:
            total = 0.0
            try:
                total = float(sum(float(value) for value in equity_map.values()))
            except Exception:  # pragma: no cover - defensive guard
                total = 0.0
            if total > 0:
                return total

        for attr in ("total_equity", "equity", "balance"):
            if hasattr(self.portfolio, attr):
                try:
                    total = float(getattr(self.portfolio, attr))
                except Exception:  # pragma: no cover - defensive guard
                    total = 0.0
                if total > 0:
                    return total

        return 0.0

    def _strategy_equity(self, strategy_name: str) -> float:
        """Return equity allocated to ``strategy_name`` with graceful fallbacks."""

        equity_map = getattr(self.portfolio, "strategy_equity", None)
        if isinstance(equity_map, Mapping):
            try:
                strategy_equity = float(equity_map.get(strategy_name, 0.0))
            except Exception:  # pragma: no cover - defensive guard
                strategy_equity = 0.0
            else:
                if strategy_equity > 0:
                    return strategy_equity

        return self._portfolio_equity()

    def _apply_atr_sizing(
        self,
        strategy_name: str,
        intents: List[OrderIntent],
        bars: Dict[str, Bar],
    ) -> List[OrderIntent]:
        sized: List[OrderIntent] = []
        strat_equity = self._strategy_equity(strategy_name)

        atr_fn = getattr(self.datafeed, "atr", None)

        for intent in intents:
            if intent.qty > 0:
                sized.append(intent)
                continue

            symbol = intent.symbol
            spot_symbol = symbol.split(":")[-1]
            bar = bars.get(symbol) or bars.get(spot_symbol)
            atr_value = None
            if callable(atr_fn):
                try:
                    atr_value = atr_fn(spot_symbol)
                except Exception:  # pragma: no cover - defensive guard
                    atr_value = None

            price = getattr(bar, "close", None) if bar is not None else None

            risk_pct = self.limits.max_position_risk_pct
            meta = getattr(intent, "meta", None)
            if isinstance(meta, Mapping):
                try:
                    override = float(meta.get("risk_pct", risk_pct))
                except Exception:  # pragma: no cover - defensive guard
                    override = risk_pct
                if override > 0:
                    risk_pct = override

            if (
                strat_equity > 0
                and risk_pct > 0
                and atr_value is not None
                and atr_value > 0
                and price is not None
                and price > 0
            ):
                qty = qty_from_risk(
                    strat_equity,
                    risk_pct,
                    atr_value,
                    self.atr_mult,
                    price,
                )
                if qty > 0:
                    sized.append(replace(intent, qty=qty))
                    continue

            sized.append(intent)

        return sized

    def _snapshot_equity(self, total_equity: float) -> None:
        self._equity_curve.append(float(total_equity))

        # Best effort attempt to keep the portfolio's own history in sync.
        if hasattr(self.portfolio, "append_equity_snapshot"):
            try:
                self.portfolio.append_equity_snapshot(total_equity)
                return
            except Exception:  # pragma: no cover - defensive guard
                pass

        internal_curve = getattr(self.portfolio, "_equity_curve", None)
        if isinstance(internal_curve, list):
            internal_curve.append(float(total_equity))
        else:
            curve = getattr(self.portfolio, "equity_curve", None)
            if isinstance(curve, list):
                curve.append(float(total_equity))

    def step(self) -> None:
        bars = self.datafeed.latest_bars()
        intents: List[OrderIntent] = []
        for strategy in self.strategies:
            try:
                raw_intents = strategy.on_bar(bars)
                intents.extend(self._apply_atr_sizing(strategy.name, raw_intents, bars))
            except Exception as exc:  # pragma: no cover - defensive logging
                self.log.exception("%s on_bar error: %s", strategy.name, exc)

        for intent in intents:
            order = self.broker.intent_to_order(intent)
            self.reconciler.submit_idempotent(order)

        total_equity = self._portfolio_equity()
        self._snapshot_equity(total_equity)

        hit, reason = self.kill.check(total_equity)
        if hit or self.reconciler.check_kill_switch(self._equity_curve):
            self.log.error(reason or "Kill-switch (reconciler) hit; stopping.")
            raise SystemExit(2)


__all__ = ["Orchestrator"]
