"""Example live trading orchestration script.

This module wires together the core engine components with ccxt exchanges
and executes a short paper-trading loop. Real API keys can be supplied via
environment variables in order to place live orders. The script is intended
as a simple starting point and may be extended with more sophisticated
scheduling, logging, or persistence as needed.
"""

from __future__ import annotations

import logging
import os
import time

import ccxt

from engine.datafeed import CCXTFeed
from engine.orchestrator import Orchestrator
from engine.portfolio import Allocation, PortfolioBook
from engine.risk import RiskCfg
from execution.adapters.ccxt_broker import CCXTBroker
from strategies.arbitrage_xex import CrossExArb
from strategies.dca_martingale import DCAMartingale
from strategies.grid import GridStrategy
from strategies.momentum_ema import MomentumEMA

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Run a short demo loop for the orchestrator."""
    # Build exchanges (keys optional for public read; needed for placing orders)
    binance = ccxt.binance(
        {
            "apiKey": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_SECRET"),
            "enableRateLimit": True,
        }
    )
    coinbase = ccxt.coinbase(
        {
            "apiKey": os.getenv("COINBASE_API_KEY"),
            "secret": os.getenv("COINBASE_SECRET"),
            "enableRateLimit": True,
        }
    )
    exchanges = {"BINANCE": binance, "COINBASE": coinbase}

    # Broker (ccxt) for one exchange to start
    broker = CCXTBroker(
        "binance",
        os.getenv("BINANCE_API_KEY"),
        os.getenv("BINANCE_SECRET"),
        testnet=False,
    )

    # Datafeed
    symbols = ["BTC/USDT", "ETH/USDT"]
    feed = CCXTFeed(exchanges, symbols, timeframe="1m")

    # Strategies
    s_grid = GridStrategy("BTC/USDT", 50000, 70000, 15, 0.001, True)
    s_mom = MomentumEMA("ETH/USDT", 12, 26, 0.01)
    s_dca = DCAMartingale("BTC/USDT", base_qty=0.0005, step_pct=2.0, max_steps=4)
    s_arb = CrossExArb("BTC/USDT", "BINANCE", "COINBASE", min_edge_bps=15, qty=0.001)
    strategies = [s_grid, s_mom, s_dca, s_arb]

    # Risk & portfolio
    risk = RiskCfg(1.0, 3.0, 8.0, 5)
    pb = PortfolioBook(
        10000.0,
        Allocation({"arbitrage": 0.25, "grid": 0.25, "momentum": 0.20, "dca": 0.30}),
    )

    orch = Orchestrator(strategies, broker, risk, pb, feed)

    # Run a short paper loop (replace with your scheduler/daemon)
    for _ in range(60):
        orch.step()
        time.sleep(5)


if __name__ == "__main__":
    main()
