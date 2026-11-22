from dataclasses import replace

import pytest

from engine.portfolio import Portfolio, StrategyAllocation, OrderFill
from engine.risk import RiskManager
from execution.adapters.ccxt_executor import CCXTSignalExecutor
from strategies.base import StrategySignal
from tradingbot_ibkr.execution.broker_base import OrderStatus
from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.execution.reconciler import Reconciler
from tradingbot_ibkr.rescue_runtime import SyntheticPaperEngine


def signal(side="buy", quantity=1, strategy="s"):
    return StrategySignal(strategy=strategy, symbol="BTC", side=side, quantity=quantity, price=100)


def test_existing_exposure_and_batch_limits_allow_reductions():
    allocation = StrategyAllocation("s", 1000, max_position_notional=100)
    portfolio = Portfolio([allocation], broker=PaperBroker())
    portfolio.apply_fills("s", [OrderFill("BTC", "buy", 1, 100)])
    risk = RiskManager({"s": allocation})
    buy, sell = signal(), signal("sell")
    decision = risk.evaluate_signals("s", [buy, sell, buy, buy], portfolio.snapshot())
    assert decision.accepted == (sell, buy)
    assert len(decision.rejected) == 2
    # An oversized sell that opens a short must still respect gross exposure.
    assert risk.evaluate_signals("s", [signal("sell", 3)], portfolio.snapshot()).rejected


def test_symbol_limit_aggregates_strategies_and_batch():
    allocations = [StrategyAllocation(name, 1000) for name in ("s", "t")]
    portfolio = Portfolio(allocations, broker=PaperBroker())
    portfolio.apply_fills("t", [OrderFill("BTC", "buy", 1, 100)])
    risk = RiskManager({a.name: a for a in allocations}, portfolio_limits={"max_symbol_notional": 150})
    assert risk.evaluate_signals("s", [signal()], portfolio.snapshot()).rejected
    small = signal(quantity=.25)
    decision = risk.evaluate_signals("s", [small, small, small], portfolio.snapshot())
    assert len(decision.accepted) == 2


def test_actual_partial_fill_prices_and_duplicate_accounting():
    class Broker:
        status = OrderStatus(None, None, "open", filled_qty=1, avg_price=110)
        def submit_order(self, order):
            return self.status
    broker = Broker()
    executor = CCXTSignalExecutor(broker, fees_bps=10)
    intent = signal(quantity=2)
    first = executor.execute(intent)
    assert first.price == 110
    assert abs(first.fee - .11) < 1e-10
    assert executor.execute(intent) is None
    assert len(executor.open_orders) == 1
    broker.status = replace(broker.status, status="filled", filled_qty=2, avg_price=120)
    second = executor.execute(intent)
    assert second.quantity == 1
    assert second.price == 130
    assert not executor.open_orders
    assert executor._to_order(signal()).id != executor._to_order(signal()).id


def test_local_reconciliation_marks_and_latched_stop():
    broker = PaperBroker()
    executor = CCXTSignalExecutor(broker)
    portfolio = Portfolio([StrategyAllocation("s", 1000)], broker=broker)
    fill = executor.execute(signal())
    portfolio.apply_fills("s", [fill])
    report = Reconciler(broker).reconcile(local_orders=executor.open_orders, local_positions=portfolio.position_quantities())
    assert report.is_clean
    assert portfolio.snapshot(mark_prices={"BTC": 120}).total_equity == 1020

    class Orchestrator:
        calls = 0
        def reconcile(self, reconciler):
            return reconciler.reconcile(local_orders={}, local_positions={})
        def run_cycle(self):
            self.calls += 1
    orchestrator = Orchestrator()
    engine = SyntheticPaperEngine(orchestrator, None, broker)
    with pytest.raises(RuntimeError, match="reconciliation mismatch"):
        engine.step()
    with pytest.raises(RuntimeError):
        engine.step()
    assert orchestrator.calls == 0
    assert engine.kill_switch_triggered
