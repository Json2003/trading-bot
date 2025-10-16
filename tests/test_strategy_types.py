"""Unit tests for the core strategy domain types."""

from tradingbot_core.strategy import Bar, OrderIntent, Strategy


class DummyStrategy:
    name = "dummy"
    symbols = ["BTCUSDT"]

    def on_bar(self, bars):
        return []

    def on_fill(self, fill):
        self.last_fill = fill

    def risk_state(self):
        return {"exposure": 0.0}


def test_order_intent_meta_isolation():
    first = OrderIntent(
        idemp_key="1",
        symbol="BTCUSDT",
        side="buy",
        qty=1.0,
        type="limit",
        limit_price=25000.0,
    )
    first.meta["note"] = "initial"

    second = OrderIntent(
        idemp_key="2",
        symbol="BTCUSDT",
        side="sell",
        qty=0.5,
        type="market",
    )

    assert second.meta == {}
    assert first.meta == {"note": "initial"}


def test_strategy_protocol_compatible():
    def drive_strategy(strategy: Strategy) -> None:
        bar = Bar(ts=1, open=10, high=11, low=9, close=10.5, volume=1000)
        intents = strategy.on_bar({"BTCUSDT": bar})
        assert intents == []
        strategy.on_fill({"symbol": "BTCUSDT", "qty": 1})
        state = strategy.risk_state()
        assert "exposure" in state

    drive_strategy(DummyStrategy())

