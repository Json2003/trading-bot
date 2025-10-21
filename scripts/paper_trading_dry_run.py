#!/usr/bin/env python3
"""Run a short paper-trading dry run using synthetic market data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.synthetic import build_synthetic_feed
from config.portfolio_loader import load_portfolio_config, load_strategy_params
from engine.builders import build_multi_strategy_orchestrator, collect_market_instruments
from engine.overlays import OverlayEngine
from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.execution.reconciler import Reconciler
from execution.adapters.ccxt_executor import CCXTSignalExecutor

LOGGER = logging.getLogger("paper.dryrun")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--portfolio-config", default="config/portfolio.yaml")
    parser.add_argument("--strategy-dir", default="config/strategy")
    parser.add_argument("--overlay-config", default="config/factors.yaml")
    parser.add_argument("--fees-bps", type=float, default=5.0)
    parser.add_argument("--slip-bps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    portfolio_cfg = load_portfolio_config(args.portfolio_config)
    strategy_params = {
        cfg.name: load_strategy_params(cfg.name, args.strategy_dir)
        for cfg in portfolio_cfg.strategies
    }

    instruments = collect_market_instruments(strategy_params, default_timeframe="1h")
    data_feed, price_history = build_synthetic_feed(
        instruments, steps=args.cycles + 5, seed=args.seed, timeframe="1h"
    )

    broker = PaperBroker()
    executor = CCXTSignalExecutor(broker, fees_bps=args.fees_bps, slippage_bps=args.slip_bps, log=LOGGER)

    overlay = OverlayEngine(price_history, args.overlay_config)

    orchestrator, portfolio, _, _, _ = build_multi_strategy_orchestrator(
        portfolio_config=portfolio_cfg,
        strategy_params=strategy_params,
        clients={},
        broker=broker,
        executor=executor,
        default_timeframe="1h",
        ohlcv_candles=120,
        log=LOGGER,
        data_feed=data_feed,
        overlay=overlay,
    )

    reconciler = Reconciler(broker)

    for cycle in range(1, args.cycles + 1):
        fills = orchestrator.run_cycle()
        snapshot = portfolio.snapshot()
        LOGGER.info("Cycle %s -> fills %d, equity %.2f", cycle, len(fills), snapshot.total_equity)

        report = reconciler.reconcile(local_orders={}, local_positions={})
        if report.is_clean:
            LOGGER.info("Reconciliation clean")
        else:
            LOGGER.warning("Reconciliation issues detected: %s", report)

        if orchestrator.kill_switch_triggered:
            event = orchestrator.kill_switch_event
            LOGGER.error("Kill-switch tripped: %s", event.reason if event else "unknown reason")
            LOGGER.info("Dispatching webhook alert (simulated)")
            break


if __name__ == "__main__":
    main()

