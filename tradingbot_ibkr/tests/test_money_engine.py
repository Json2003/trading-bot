import math


from tradingbot_ibkr import money_engine


def test_qty_from_risk_happy_path():
    qty = money_engine.qty_from_risk(
        equity=10_000.0,
        risk_pct=1.0,
        atr=50.0,
        atr_mult=2.0,
        price=25_000.0,
    )

    assert math.isclose(qty, 1.0)


def test_qty_from_risk_invalid_inputs():
    assert money_engine.qty_from_risk(0, 1.0, 10.0, 2.0, 30_000.0) == 0.0
    assert money_engine.qty_from_risk(10_000.0, 0.0, 10.0, 2.0, 30_000.0) == 0.0
    assert money_engine.qty_from_risk(10_000.0, 1.0, 0.0, 2.0, 30_000.0) == 0.0
    assert money_engine.qty_from_risk(10_000.0, 1.0, 10.0, 0.0, 30_000.0) == 0.0
    assert money_engine.qty_from_risk(10_000.0, 1.0, 10.0, 2.0, 0.0) == 0.0
