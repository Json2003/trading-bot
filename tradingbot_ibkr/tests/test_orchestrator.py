from __future__ import annotations

from math import isclose
from types import SimpleNamespace

from tradingbot_core.strategy import OrderIntent

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
    def __init__(self, bars):
        self._bars = bars

    def latest_bars(self):
        return self._bars


class DummyPortfolio:
    def __init__(self, equity_curve):
        self.equity_curve = equity_curve


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
