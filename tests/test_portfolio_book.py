from math import isclose

import pytest

from models import Allocation, PortfolioBook


def test_allocation_sum_to_one_validation():
    with pytest.raises(ValueError):
        Allocation({"arbitrage": 0.4, "grid": 0.5})


def test_allocation_negative_percentage_validation():
    with pytest.raises(ValueError):
        Allocation({"arbitrage": -0.1, "grid": 1.1})


def test_portfolio_initializes_strategy_equity_and_curve():
    alloc = Allocation({"arbitrage": 0.6, "grid": 0.4})
    book = PortfolioBook(10_000, alloc)

    assert isclose(book.total_equity, 10_000)
    assert isclose(book.strategy_equity["arbitrage"], 6000)
    assert isclose(book.strategy_equity["grid"], 4000)
    assert len(book.equity_curve) == 1
    assert isclose(book.equity_curve[0], 10_000)


def test_credit_pnl_updates_strategy_equity_and_curve():
    alloc = Allocation({"arbitrage": 0.5, "grid": 0.5})
    book = PortfolioBook(8_000, alloc)

    book.credit_pnl("arbitrage", 500)

    assert isclose(book.strategy_equity["arbitrage"], 4_500)
    assert isclose(book.total_equity, 8_500)
    assert len(book.equity_curve) == 2
    assert isclose(book.equity_curve[0], 8_000)
    assert isclose(book.equity_curve[1], 8_500)


def test_credit_pnl_unknown_strategy():
    alloc = Allocation({"arbitrage": 0.5, "grid": 0.5})
    book = PortfolioBook(5_000, alloc)

    with pytest.raises(KeyError):
        book.credit_pnl("momentum", 100)
