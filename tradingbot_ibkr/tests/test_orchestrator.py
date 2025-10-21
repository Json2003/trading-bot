from __future__ import annotations

from math import isclose, log
from types import SimpleNamespace
from typing import Mapping
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
        self.submitted_orders: list[dict[str, object]] = []

    def intent_to_order(self, intent: OrderIntent):
        self.received.append(intent)
        return {"id": intent.idemp_key, "symbol": intent.symbol}

    def submit_order(self, **order):
        self.submitted_orders.append(order)
        return order


class DummyDataFeed:
    def __init__(self, bars, atr_value: float | None = None):
        self._bars = bars
        self._atr_value = atr_value

    def latest_bars(self):
        return self._bars

    def atr(self, symbol):  # pragma: no cover - only used in specific tests
        return self._atr_value


class DummyPortfolio:
    def __init__(
        self,
        equity_curve,
        *,
        strategy_equity=None,
        total_equity=None,
        alloc=None,
        exposures=None,
    ):
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

        if alloc is None:
            weights: dict[str, float] = {}
            if self.strategy_equity:
                total = sum(self.strategy_equity.values()) or 1.0
                weights = {name: value / total for name, value in self.strategy_equity.items()}
            alloc = SimpleNamespace(per_strategy_pct=weights)
        self.alloc = alloc

        self._exposures = exposures or {}

    def current_exposures_quote_currency(self):
        return dict(self._exposures)


class CyclingDataFeed:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        if not self._snapshots:
            raise ValueError("snapshots must be non-empty")
        self._index = 0

    def latest_bars(self):
        bars = self._snapshots[self._index]
        if self._index < len(self._snapshots) - 1:
            self._index += 1
        return bars


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


class RecordingBeta:
    def __init__(self, latest=None):
        self.updates: list[tuple[str, float, float]] = []
        self.latest = latest or {}

    def update(self, symbol, r_symbol, r_market):
        self.updates.append((symbol, r_symbol, r_market))


def test_step_updates_beta_with_log_returns():
    beta = RecordingBeta(latest={"ETH/USDT": 1.0})
    snapshots = [
        {
            "BTC/USDT": Bar(ts=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
            "ETH/USDT": Bar(ts=1, open=50.0, high=50.0, low=50.0, close=50.0, volume=1.0),
        },
        {
            "BTC/USDT": Bar(ts=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
            "ETH/USDT": Bar(ts=2, open=55.0, high=55.0, low=55.0, close=55.0, volume=1.0),
        },
    ]
    feed = CyclingDataFeed(snapshots)

    risk_cfg = SimpleNamespace(
        max_daily_loss_pct=3.0,
        kill_switch_drawdown_pct=8.0,
        per_trade_risk_pct=1.5,
    )

    orchestrator = orchestrator_module.Orchestrator(
        strategies=[DummyStrategy([])],
        broker=DummyBroker(),
        risk_cfg=risk_cfg,
        portfolio_book=DummyPortfolio([1000.0], strategy_equity={"dummy": 1000.0}),
        datafeed=feed,
        tradable_symbols=["ETH/USDT"],
        beta=beta,
        rebalance_k=0,
    )

    orchestrator.step()
    orchestrator.step()

    assert beta.updates, "beta updates should be recorded"
    symbol, r_symbol, r_market = beta.updates[-1]
    assert symbol == "ETH/USDT"
    assert isclose(r_symbol, log(55.0 / 50.0))
    assert isclose(r_market, log(110.0 / 100.0))


class PassiveBeta:
    def __init__(self):
        self.latest = {}

    def update(self, symbol, *_):  # pragma: no cover - no behaviour required
        pass


def test_component_momentum_tilts_allocations(monkeypatch):
    beta = PassiveBeta()
    comp_cfg = SimpleNamespace(lookback=3, skip=0, tilt_strength=0.5)
    cfg = SimpleNamespace(comp_m=comp_cfg)
    snapshots = [
        {
            "BTC/USDT": Bar(ts=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
            "A/USDT": Bar(ts=1, open=10.0, high=10.0, low=10.0, close=10.0, volume=1.0),
            "B/USDT": Bar(ts=1, open=10.0, high=10.0, low=10.0, close=10.0, volume=1.0),
        },
        {
            "BTC/USDT": Bar(ts=2, open=101.0, high=101.0, low=101.0, close=101.0, volume=1.0),
            "A/USDT": Bar(ts=2, open=10.5, high=10.5, low=10.5, close=10.5, volume=1.0),
            "B/USDT": Bar(ts=2, open=9.8, high=9.8, low=9.8, close=9.8, volume=1.0),
        },
        {
            "BTC/USDT": Bar(ts=3, open=102.0, high=102.0, low=102.0, close=102.0, volume=1.0),
            "A/USDT": Bar(ts=3, open=11.0, high=11.0, low=11.0, close=11.0, volume=1.0),
            "B/USDT": Bar(ts=3, open=9.7, high=9.7, low=9.7, close=9.7, volume=1.0),
        },
    ]
    feed = CyclingDataFeed(snapshots)

    alloc = SimpleNamespace(per_strategy_pct={"A/USDT": 0.5, "B/USDT": 0.5})
    portfolio = DummyPortfolio(
        [1000.0],
        strategy_equity={"A/USDT": 500.0, "B/USDT": 500.0},
        alloc=alloc,
    )

    risk_cfg = SimpleNamespace(
        max_daily_loss_pct=3.0,
        kill_switch_drawdown_pct=8.0,
        per_trade_risk_pct=1.5,
    )

    orchestrator = orchestrator_module.Orchestrator(
        strategies=[DummyStrategy([])],
        broker=DummyBroker(),
        risk_cfg=risk_cfg,
        portfolio_book=portfolio,
        datafeed=feed,
        tradable_symbols=["A/USDT", "B/USDT"],
        beta=beta,
        cfg=cfg,
        rebalance_k=2,
    )

    orchestrator.step()
    orchestrator.step()
    orchestrator.step()

    updated_alloc = portfolio.alloc.per_strategy_pct
    assert isclose(sum(updated_alloc.values()), 1.0, rel_tol=1e-6)
    assert updated_alloc["A/USDT"] > 0.5
    assert updated_alloc["B/USDT"] < 0.5


class RecordingHedger:
    def __init__(self, notional):
        self.notional = notional
        self.calls: list[tuple[dict[str, float], Mapping[str, float], float]] = []

    def hedge_notional(self, exposures, betas, *, total_equity):
        self.calls.append((dict(exposures), dict(betas), float(total_equity)))
        return self.notional


def test_beta_hedge_submits_market_order_when_notional_large():
    beta = RecordingBeta(latest={"ETH/USDT": 1.2})
    hedger = RecordingHedger(500.0)
    exposures = {"ETH/USDT": 200.0}
    portfolio = DummyPortfolio(
        [1000.0],
        strategy_equity={"dummy": 1000.0},
        exposures=exposures,
    )
    snapshots = [
        {
            "BTC/USDT": Bar(ts=1, open=25_000.0, high=25_000.0, low=25_000.0, close=25_000.0, volume=1.0),
            "ETH/USDT": Bar(ts=1, open=1_500.0, high=1_500.0, low=1_500.0, close=1_500.0, volume=1.0),
        }
    ]
    feed = CyclingDataFeed(snapshots)

    risk_cfg = SimpleNamespace(
        max_daily_loss_pct=3.0,
        kill_switch_drawdown_pct=8.0,
        per_trade_risk_pct=1.5,
    )

    broker = DummyBroker()
    orchestrator = orchestrator_module.Orchestrator(
        strategies=[DummyStrategy([])],
        broker=broker,
        risk_cfg=risk_cfg,
        portfolio_book=portfolio,
        datafeed=feed,
        tradable_symbols=["ETH/USDT"],
        beta=beta,
        hedger=hedger,
        rebalance_k=1,
        min_notional=100.0,
    )

    orchestrator.step()

    assert hedger.calls, "hedger should be invoked"
    assert broker.submitted_orders, "beta hedge order should be submitted"
    order = broker.submitted_orders[-1]
    assert order["symbol"] == "BTC/USDT"
    assert order["side"] == "sell"
    assert isclose(order["qty"], 500.0 / 25_000.0)
