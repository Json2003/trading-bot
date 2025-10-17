"""Execution orchestrator wiring strategies, broker and reconciler together."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from statistics import mean, pstdev
from typing import Dict, List
import logging
import math

import numpy as np

from tradingbot_core.risk import KillSwitch, KillSwitchCfg
from tradingbot_core.strategy import Bar, OrderIntent, Strategy

from tradingbot_ibkr.execution.reconciler import Reconciler, RiskLimits
from tradingbot_ibkr.money_engine import qty_from_risk


def comp_m_scores(
    price_history: Mapping[str, np.ndarray], *, lookback: int, skip: int = 0
) -> dict[str, float]:
    """Compute momentum z-scores for each symbol based on log returns."""

    scores: dict[str, float] = {}
    if lookback <= 1:
        return scores

    for symbol, prices in price_history.items():
        if prices.size == 0:
            continue

        series = np.asarray(prices, dtype=float)
        if skip > 0 and series.size > skip:
            series = series[:-skip]
        elif series.size <= skip:
            continue

        if series.size < lookback:
            continue

        window = series[-lookback:]
        window_list = [float(value) for value in window]
        log_returns = [
            math.log(window_list[idx] / window_list[idx - 1])
            for idx in range(1, len(window_list))
            if window_list[idx - 1] > 0 and window_list[idx] > 0
        ]
        if not log_returns:
            continue

        mean_value = mean(log_returns)
        std_value = pstdev(log_returns) if len(log_returns) > 1 else 0.0
        if std_value <= 1e-12:
            continue

        scores[symbol] = float(mean_value / std_value)

    return scores


def tilt_allocations(
    allocations: Mapping[str, float], symbol: str, tilt: float
) -> dict[str, float]:
    """Return updated allocations with ``symbol`` tilted by ``tilt`` factor."""

    if not allocations:
        return {}

    adjusted = dict(allocations)
    if symbol not in adjusted:
        return adjusted

    factor = float(max(tilt, 0.0))
    adjusted[symbol] = adjusted[symbol] * factor
    total = sum(adjusted.values())
    if total <= 0:
        return dict(allocations)

    return {name: value / total for name, value in adjusted.items()}


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
        tradable_symbols: Sequence[str] | None = None,
        *,
        beta=None,
        hedger=None,
        cfg=None,
        rebalance_k: int | None = None,
        min_notional: float | None = None,
    ) -> None:
        self.strategies = strategies
        self.broker = broker
        self.portfolio = portfolio_book
        self.datafeed = datafeed
        self.atr_mult = float(atr_mult)
        self.log = logger or logging.getLogger("orchestrator")

        if tradable_symbols is None:
            derived: list[str] = []
            for strategy in strategies:
                symbols = getattr(strategy, "symbols", None)
                if isinstance(symbols, Sequence):
                    derived.extend(str(symbol) for symbol in symbols)
            self.tradable_symbols = sorted(set(derived))
        else:
            self.tradable_symbols = [str(symbol) for symbol in tradable_symbols]

        self.beta = beta
        self.hedger = hedger
        self.cfg = cfg

        if rebalance_k is None and cfg is not None:
            rebalance_k = getattr(cfg, "rebalance_K", None)
            if rebalance_k is None:
                rebalance_k = getattr(cfg, "rebalance_k", None)
        self._rebalance_k = int(rebalance_k) if rebalance_k else 0

        cfg_min_notional = 0.0
        if cfg is not None:
            cfg_min_notional = float(getattr(cfg, "min_notional", 0.0) or 0.0)
        self._min_notional = float(min_notional) if min_notional is not None else cfg_min_notional

        comp_cfg = getattr(cfg, "comp_m", None)
        lookback = int(getattr(comp_cfg, "lookback", 0) or 0) if comp_cfg else 0
        skip = int(getattr(comp_cfg, "skip", 0) or 0) if comp_cfg else 0
        self._history_length = max(lookback + skip + 2, 0)

        self._last_price: dict[str, float] = {}
        self._last_market_price: float | None = None
        self._hist_prices: dict[str, list[float]] = defaultdict(list)
        self._bar_index = 0
        self._market_symbol = "BTC/USDT"

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

        bar_index = self._bar_index
        self._update_log_returns(bars)
        self._maybe_rebalance_allocations(bar_index)
        self._maybe_beta_hedge(bar_index, bars)

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

        self._bar_index = bar_index + 1

    def _update_log_returns(self, bars: Mapping[str, Bar]) -> None:
        if not self.tradable_symbols or self.beta is None:
            return

        market_bar = bars.get(self._market_symbol)
        market_close = getattr(market_bar, "close", None) if market_bar else None
        last_market_price = self._last_market_price
        if market_close is None or market_close <= 0:
            market_return = 0.0
        elif last_market_price is None or last_market_price <= 0:
            market_return = 0.0
        else:
            market_return = math.log(market_close / last_market_price)

        if market_close is not None and market_close > 0:
            self._last_market_price = float(market_close)

        for symbol in self.tradable_symbols:
            bar = bars.get(symbol)
            price = getattr(bar, "close", None) if bar else None
            if price is None or price <= 0:
                continue

            prev_price = self._last_price.get(symbol, price)
            symbol_return = math.log(price / prev_price) if prev_price > 0 else 0.0
            self._last_price[symbol] = float(price)

            history = self._hist_prices[symbol]
            history.append(float(price))
            if self._history_length and len(history) > self._history_length:
                del history[0 : len(history) - self._history_length]

            try:
                self.beta.update(symbol, symbol_return, market_return)
            except Exception:  # pragma: no cover - defensive guard
                self.log.exception("Failed to update beta for %s", symbol)

    def _maybe_rebalance_allocations(self, bar_index: int) -> None:
        if self._rebalance_k <= 0:
            return

        if bar_index % self._rebalance_k != 0:
            return

        comp_cfg = getattr(self.cfg, "comp_m", None)
        if comp_cfg is None:
            return

        lookback = int(getattr(comp_cfg, "lookback", 0) or 0)
        skip = int(getattr(comp_cfg, "skip", 0) or 0)
        tilt_strength = float(getattr(comp_cfg, "tilt_strength", 0.0) or 0.0)

        price_hist = {
            symbol: np.asarray(self._hist_prices.get(symbol, []), dtype=float)
            for symbol in self.tradable_symbols
        }
        scores = comp_m_scores(price_hist, lookback=lookback, skip=skip)
        if not scores:
            return

        alloc = getattr(self.portfolio, "alloc", None)
        if alloc is None or not hasattr(alloc, "per_strategy_pct"):
            return

        current_alloc = getattr(alloc, "per_strategy_pct")
        if not isinstance(current_alloc, Mapping):
            return

        updated = dict(current_alloc)
        changed = False
        for symbol, score in scores.items():
            if symbol not in updated:
                continue
            tilt = np.clip(1 + tilt_strength * score, 0.2, 1.8)
            updated = tilt_allocations(updated, symbol, float(tilt))
            changed = True

        if changed:
            setattr(alloc, "per_strategy_pct", updated)

    def _maybe_beta_hedge(self, bar_index: int, bars: Mapping[str, Bar]) -> None:
        if self._rebalance_k <= 0 or bar_index % self._rebalance_k != 0:
            return

        if self.beta is None or self.hedger is None:
            return

        exposures_fn = getattr(self, "current_exposures_quote_currency", None)
        if exposures_fn is None:
            exposures_fn = getattr(self.portfolio, "current_exposures_quote_currency", None)

        exposures = {}
        if callable(exposures_fn):
            try:
                exposures = exposures_fn() or {}
            except Exception:  # pragma: no cover - defensive guard
                self.log.exception("Failed to obtain current exposures")
                exposures = {}

        betas = getattr(self.beta, "latest", None)
        if not isinstance(betas, Mapping) or not betas:
            return

        total_equity = self._portfolio_equity()
        try:
            notional = self.hedger.hedge_notional(exposures, betas, total_equity=total_equity)
        except Exception:  # pragma: no cover - defensive guard
            self.log.exception("Beta hedging notional calculation failed")
            return

        if not isinstance(notional, (int, float)):
            return

        notional_value = float(notional)
        if abs(notional_value) <= self._min_notional:
            return

        market_bar = bars.get(self._market_symbol)
        market_price = getattr(market_bar, "close", None) if market_bar else None
        if market_price is None or market_price <= 0:
            return

        qty = abs(notional_value) / float(market_price)
        side = "sell" if notional_value > 0 else "buy"

        submit_fn = getattr(self.broker, "submit_order", None)
        if not callable(submit_fn):
            return

        try:
            submit_fn(
                symbol=self._market_symbol,
                side=side,
                qty=qty,
                type="market",
                tag="beta-hedge",
            )
        except Exception:  # pragma: no cover - defensive guard
            self.log.exception("Beta hedge order submission failed")


__all__ = ["Orchestrator"]
