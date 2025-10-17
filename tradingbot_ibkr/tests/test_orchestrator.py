from __future__ import annotations

from math import isclose
from types import SimpleNamespace

from tradingbot_core.strategy import Bar, OrderIntent

import tradingbot_ibkr.orchestrator as orchestrator_module


class DummyStrategy:
    name = "dummy"
    symbols = ["BTC/USDT"]

    def __init__(self, intents):
        self._intents = intents

    def on_bar(self, bars):
        return list(self._intents)

    def on_fill(self, fill):  # pragma: no cover - not exercised
        pass

    def risk_state(self):  # pragma: no cover - not exercised
        return {}


class DummyBroker:
    def __init__(self) -> None:
        self.received: list[OrderIntent] = []

    def intent_to_order(self, intent: OrderIntent):
        self.received.append(intent)
        return {"id": intent.idemp_key, "symbol": intent.symbol}


class DummyDataFeed:
    def __init__(self, bars, atr_value: float | None = None):
        self._bars = bars
        self._atr_value = atr_value

    def latest_bars(self):
        return self._bars

    def atr(self, symbol):  # pragma: no cover - only used in specific tests
        return self._atr_value


class DummyPortfolio:
    def __init__(self, equity_curve, *, strategy_equity=None, total_equity=None):
        self.equity_curve = equity_curve

        if strategy_equity is None:
            last = float(equity_curve[-1]) if equity_curve else 0.0
            self.strategy_equity = {"dummy": last} if last else {}
        else:
            self.strategy_equity = strategy_equity

        if total_equity is None and equity_curve:
            total_equity = float(equity_curve[-1])
        if total_equity is not None:
            self.total_equity = total_equity


class RecordingReconciler:
    def __init__(self, broker, *, limits, logger):
        self.broker = broker
        self.limits = limits
        self.logger = logger
        self.submitted = []
        self.checked_equity = None

    def submit_idempotent(self, order):
        self.submitted.append(order)
        return order

    def check_kill_switch(self, equity_curve):
        self.checked_equity = equity_curve
        return False


def test_orchestrator_configures_reconciler_with_risk_limits(monkeypatch):
    intents = [
        OrderIntent(
            idemp_key="k1",
            symbol="BTC/USDT",
            side="buy",
            qty=1.0,
            type="market",
        )
    ]
    strategy = DummyStrategy(intents)
    broker = DummyBroker()
    datafeed = DummyDataFeed({"BTC/USDT": object()})
    portfolio = DummyPortfolio([1000.0, 1010.0])

    created: list[RecordingReconciler] = []

    def factory(*args, **kwargs):
        instance = RecordingReconciler(*args, **kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(orchestrator_module, "Reconciler", factory)

    risk_cfg = SimpleNamespace(
        max_daily_loss_pct=3.0,
        kill_switch_drawdown_pct=8.0,
        per_trade_risk_pct=1.5,
    )

    orchestrator = orchestrator_module.Orchestrator(
        strategies=[strategy],
        broker=broker,
        risk_cfg=risk_cfg,
        portfolio_book=portfolio,
        datafeed=datafeed,
    )

    assert created, "Reconciler should be instantiated"
    reconciler = created[0]

    assert isinstance(reconciler.limits, orchestrator_module.RiskLimits)
    assert isclose(reconciler.limits.max_daily_loss_pct, risk_cfg.max_daily_loss_pct)
    assert isclose(reconciler.limits.kill_switch_drawdown_pct, risk_cfg.kill_switch_drawdown_pct)
    assert isclose(reconciler.limits.max_position_risk_pct, risk_cfg.per_trade_risk_pct)
    assert reconciler.logger is orchestrator.log

    orchestrator.step()

    assert broker.received == intents
    assert reconciler.submitted == [{"id": "k1", "symbol": "BTC/USDT"}]
    assert reconciler.checked_equity == portfolio.equity_curve


def test_orchestrator_sizes_zero_qty_intents_using_atr(monkeypatch):
    intent = OrderIntent(
        idemp_key="risk-sized",
        symbol="BINANCE:BTC/USDT",
        side="buy",
        qty=0.0,
        type="market",
        meta={"risk_pct": 0.5},
    )
    strategy = DummyStrategy([intent])
    broker = DummyBroker()

    bar = Bar(ts=1, open=100.0, high=110.0, low=90.0, close=100.0, volume=1_000.0)
    datafeed = DummyDataFeed({"BINANCE:BTC/USDT": bar, "BTC/USDT": bar}, atr_value=10.0)
    portfolio = DummyPortfolio([1000.0, 1010.0], strategy_equity={}, total_equity=20_000.0)

    created: list[RecordingReconciler] = []

    def factory(*args, **kwargs):
        instance = RecordingReconciler(*args, **kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(orchestrator_module, "Reconciler", factory)

    risk_cfg = SimpleNamespace(
        max_daily_loss_pct=3.0,
        kill_switch_drawdown_pct=8.0,
        per_trade_risk_pct=1.0,
    )

    orchestrator = orchestrator_module.Orchestrator(
        strategies=[strategy],
        broker=broker,
        risk_cfg=risk_cfg,
        portfolio_book=portfolio,
        datafeed=datafeed,
    )

    orchestrator.step()

    assert created, "Reconciler should be instantiated"
    sized_intent = broker.received[0]

    expected_qty = orchestrator_module.qty_from_risk(20_000.0, 0.5, 10.0, orchestrator.atr_mult, 100.0)
    assert isclose(sized_intent.qty, expected_qty)
    assert sized_intent.qty > 0
