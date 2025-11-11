#!/usr/bin/env python3
"""Run the multi-strategy portfolio using synthetic market data."""

from __future__ import annotations

import argparse
import json
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
from tradingbot_core.backtest_harness import BacktestContext, BacktestHarness
from tradingbot_ibkr.execution.paper_broker import PaperBroker
from execution.adapters.ccxt_executor import CCXTSignalExecutor

LOGGER = logging.getLogger("portfolio.backtest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-config", default="config/portfolio.yaml")
    parser.add_argument("--strategy-dir", default="config/strategy")
    parser.add_argument("--cycles", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fees-bps", type=float, default=10.0)
    parser.add_argument("--slip-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--filename-prefix", default="backtest_multi_strategy")
    parser.add_argument("--enable-sizing", action="store_true")
    parser.add_argument("--enable-kill-switch", action="store_true")
    parser.add_argument("--enable-overlays", action="store_true")
    parser.add_argument("--overlay-config", default="config/factors.yaml")
    return parser.parse_args()


def run_backtest(args: argparse.Namespace) -> dict[str, object]:
    portfolio_cfg = load_portfolio_config(args.portfolio_config)
    strategy_params = {
        cfg.name: load_strategy_params(cfg.name, args.strategy_dir)
        for cfg in portfolio_cfg.strategies
    }

    instruments = collect_market_instruments(strategy_params, default_timeframe="1h")
    data_feed, price_history = build_synthetic_feed(
        instruments, steps=args.cycles, seed=args.seed, timeframe="1h"
    )

    broker = PaperBroker()
    executor = CCXTSignalExecutor(broker, fees_bps=args.fees_bps, slippage_bps=args.slip_bps, log=LOGGER)

    overlay_engine = None
    if args.enable_overlays:
        overlay_engine = OverlayEngine(price_history, args.overlay_config)

    orchestrator, portfolio, _, _, _ = build_multi_strategy_orchestrator(
        portfolio_config=portfolio_cfg,
        strategy_params=strategy_params,
        clients={},
        broker=broker,
        executor=executor,
        default_timeframe="1h",
        ohlcv_candles=90,
        log=LOGGER,
        data_feed=data_feed,
        enable_sizing=args.enable_sizing,
        enable_kill_switch=args.enable_kill_switch,
        overlay=overlay_engine,
    )

    equity_curve = [portfolio.snapshot().total_equity]
    returns: list[float] = []
    trades: list[dict[str, object]] = []

    kill_event = None
    while True:
        try:
            fills = orchestrator.run_cycle()
        except StopIteration:
            break

        market = orchestrator.last_market_data or {}
        mark_prices = {data.symbol: data.price for data in market.values()}
        snapshot = portfolio.snapshot(mark_prices=mark_prices)
        equity_curve.append(snapshot.total_equity)
        returns.append(equity_curve[-1] / equity_curve[-2] - 1.0)

        for fill in fills:
            trades.append({
                "symbol": fill.symbol,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "fee": fill.fee,
            })

        if orchestrator.kill_switch_triggered:
            kill_event = orchestrator.kill_switch_event
            break

    payload: dict[str, object] = {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "metadata": {
            "cycles": len(equity_curve) - 1,
            "sizing_enabled": args.enable_sizing,
            "kill_switch_enabled": args.enable_kill_switch,
            "overlays_enabled": args.enable_overlays,
        },
    }
    if kill_event is not None:
        payload["kill_switch"] = {
            "reason": kill_event.reason,
            "equity": kill_event.equity,
            "timestamp": kill_event.timestamp.isoformat(),
        }
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()

    harness = BacktestHarness(
        output_dir=Path(args.output_dir),
        filename_prefix=args.filename_prefix,
        metadata=BacktestContext(
            strategy="multi-strategy-portfolio",
            market="synthetic",
            timeframe="1h",
            parameters={
                "cycles": args.cycles,
                "sizing": args.enable_sizing,
                "kill_switch": args.enable_kill_switch,
            },
        ),
    )

    result_path = harness.run(lambda: run_backtest(args))
    print(json.dumps({"result_path": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()

