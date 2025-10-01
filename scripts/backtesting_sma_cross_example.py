#!/usr/bin/env python3
"""Simple moving-average crossover example using the `backtesting` package.

This script mirrors the canonical example from the ``backtesting`` project and
serves as a quick sanity check that the environment can execute the Strategy
API.  It relies on the bundled ``GOOG`` sample data, so it is completely
self-contained.
"""

from __future__ import annotations

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG, SMA


class SmaCross(Strategy):
    """Classic 50/200 simple moving-average crossover strategy."""

    n1 = 50  # Short SMA window
    n2 = 200  # Long SMA window

    def init(self) -> None:
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self) -> None:
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()  # Assumes shorting allowed; adjust for long-only


def main() -> None:
    """Execute the backtest and print the resulting performance metrics."""

    bt = Backtest(GOOG, SmaCross, cash=10_000, commission=0.002)
    stats = bt.run()
    print(stats)

    try:
        bt.plot()
    except Exception as exc:  # pragma: no cover - plotting can fail headless
        print("Plotting failed (likely due to headless environment):", exc)


if __name__ == "__main__":
    main()
