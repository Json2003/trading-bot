import math

import pytest

from tradingbot_core.strategy import Bar
from tradingbot_core.strategies import GridConfig, GridStrategy


@pytest.fixture
def sample_bar() -> Bar:
    return Bar(ts=0, open=0, high=0, low=0, close=150, volume=0)


def test_arithmetic_grid_orders(sample_bar: Bar) -> None:
    config = GridConfig(symbol="BTC/USDT", lower=100, upper=200, levels=5, quantity=0.5, geometric=False)
    strategy = GridStrategy(config)

    intents = strategy.on_bar({"BTC/USDT": sample_bar})
    assert {intent.side for intent in intents} == {"buy", "sell"}

    sells = [intent for intent in intents if intent.side == "sell"]
    buys = [intent for intent in intents if intent.side == "buy"]

    assert [intent.limit_price for intent in sells] == [100.0, 125.0]
    assert [intent.limit_price for intent in buys] == [175.0, 200.0]
    assert all(intent.qty == 0.5 for intent in intents)
    assert all(intent.type == "limit" for intent in intents)


def test_geometric_grid_prices() -> None:
    config = GridConfig(symbol="ETH/USDT", lower=100, upper=800, levels=4, quantity=1, geometric=True)
    strategy = GridStrategy(config)

    ratio = (config.upper / config.lower) ** (1 / (config.levels - 1))
    expected = [config.lower * ratio**i for i in range(config.levels)]
    for actual, target in zip(strategy.prices, expected, strict=True):
        assert math.isclose(actual, target, rel_tol=1e-9)


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        GridStrategy(GridConfig(symbol="SOL/USDT", lower=10, upper=5, levels=3, quantity=1))
    with pytest.raises(ValueError):
        GridStrategy(GridConfig(symbol="SOL/USDT", lower=10, upper=20, levels=1, quantity=1))
    with pytest.raises(ValueError):
        GridStrategy(GridConfig(symbol="SOL/USDT", lower=-10, upper=20, levels=3, quantity=1, geometric=True))
    with pytest.raises(ValueError):
        GridStrategy(GridConfig(symbol="SOL/USDT", lower=10, upper=20, levels=3, quantity=0))


def test_risk_state_contains_summary(sample_bar: Bar) -> None:
    config = GridConfig(symbol="BTC/USDT", lower=100, upper=200, levels=5, quantity=0.5)
    strategy = GridStrategy(config)

    state = strategy.risk_state()
    assert state == {"symbol": "BTC/USDT", "levels": 5, "quantity": 0.5}
