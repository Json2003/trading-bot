from __future__ import annotations

from dataclasses import dataclass
import math

from engine.portfolio import OrderFill, Portfolio, StrategyAllocation
from tradingbot_ibkr.execution.broker_base import BrokerBase, Position


@dataclass
class _StubBroker(BrokerBase):
    """Minimal broker stub exposing mutable positions for testing."""

    positions: list[Position] | None = None

    def list_open_orders(self):  # pragma: no cover - unused in tests
        return []

    def list_positions(self):
        return list(self.positions or [])


def _make_portfolio(capital: float = 1_000.0) -> tuple[Portfolio, _StubBroker]:
    broker = _StubBroker([])
    allocation = StrategyAllocation(name="alpha", capital=capital, max_position_notional=capital)
    return Portfolio([allocation], broker=broker), broker


def test_apply_fills_tracks_unrealised_and_equity():
    portfolio, broker = _make_portfolio(1_000.0)
    fill = OrderFill(symbol="BTC/USDT", side="buy", quantity=1.0, price=100.0)

    portfolio.apply_fills("alpha", [fill])
    broker.positions = [Position(symbol="BTC/USDT", quantity=1.0, average_price=100.0)]

    snapshot = portfolio.snapshot(mark_prices={"BTC/USDT": 110.0})
    state = snapshot.state_for("alpha")

    assert math.isclose(state.cash, 900.0, rel_tol=1e-9)
    assert math.isclose(state.pnl.realised, 0.0, abs_tol=1e-9)
    assert math.isclose(state.pnl.unrealised, 10.0, abs_tol=1e-9)
    assert len(state.positions) == 1
    assert math.isclose(state.positions[0].quantity, 1.0, abs_tol=1e-9)
    assert math.isclose(snapshot.total_equity, 1_010.0, abs_tol=1e-9)


def test_apply_fills_realised_pnl_on_close():
    portfolio, broker = _make_portfolio(1_000.0)
    buy = OrderFill(symbol="ETH/USDT", side="buy", quantity=1.0, price=100.0)
    sell = OrderFill(symbol="ETH/USDT", side="sell", quantity=1.0, price=120.0)

    portfolio.apply_fills("alpha", [buy])
    broker.positions = [Position(symbol="ETH/USDT", quantity=1.0, average_price=100.0)]
    portfolio.apply_fills("alpha", [sell])
    broker.positions = []

    snapshot = portfolio.snapshot()
    state = snapshot.state_for("alpha")

    assert math.isclose(state.cash, 1_020.0, abs_tol=1e-9)
    assert math.isclose(state.pnl.realised, 20.0, abs_tol=1e-9)
    assert math.isclose(state.pnl.unrealised, 0.0, abs_tol=1e-9)
    assert not state.positions
    assert math.isclose(snapshot.total_equity, 1_020.0, abs_tol=1e-9)


def test_partial_close_retains_cost_basis():
    portfolio, broker = _make_portfolio(1_000.0)
    buy1 = OrderFill(symbol="SOL/USDT", side="buy", quantity=1.0, price=100.0)
    buy2 = OrderFill(symbol="SOL/USDT", side="buy", quantity=1.0, price=120.0)
    sell = OrderFill(symbol="SOL/USDT", side="sell", quantity=1.0, price=130.0)

    portfolio.apply_fills("alpha", [buy1, buy2])
    broker.positions = [Position(symbol="SOL/USDT", quantity=2.0, average_price=110.0)]
    portfolio.apply_fills("alpha", [sell])
    broker.positions = [Position(symbol="SOL/USDT", quantity=1.0, average_price=110.0)]

    snapshot = portfolio.snapshot(mark_prices={"SOL/USDT": 130.0})
    state = snapshot.state_for("alpha")

    assert math.isclose(state.cash, 910.0, abs_tol=1e-9)
    assert math.isclose(state.pnl.realised, 20.0, abs_tol=1e-9)
    assert math.isclose(state.pnl.unrealised, 20.0, abs_tol=1e-9)
    assert len(state.positions) == 1
    assert math.isclose(state.positions[0].quantity, 1.0, abs_tol=1e-9)
    assert math.isclose(snapshot.total_equity, 1_040.0, abs_tol=1e-9)
