#!/usr/bin/env python3
"""CLI entry point for running the multi-strategy portfolio engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config.portfolio_loader import load_portfolio_config, load_strategy_params
from engine.builders import (
    build_multi_strategy_orchestrator,
    collect_market_instruments,
)
from execution.adapters.ccxt_executor import CCXTSignalExecutor
from tradingbot_ibkr.execution.paper_broker import PaperBroker

LOGGER = logging.getLogger("portfolio.runner")


def _build_ccxt_clients(venues: list[str]) -> dict[str, object]:
    try:
        import ccxt  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "ccxt is required to run the multi-strategy portfolio runner"
        ) from exc

    clients: dict[str, object] = {}
    for venue in venues:
        factory = getattr(ccxt, venue, None) or getattr(ccxt, venue.lower(), None)
        if factory is None:
            raise SystemExit(f"ccxt does not provide an exchange named {venue!r}")
        clients[venue] = factory({"enableRateLimit": True})
    return clients


def _determine_venues(strategy_params: dict[str, dict[str, object]], default_timeframe: str) -> list[str]:
    instruments = collect_market_instruments(strategy_params, default_timeframe=default_timeframe)
    venues = sorted({instrument.venue for instrument in instruments})
    return venues


def _dump_cycle_result(path: Path, cycle: int, fills, snapshot) -> None:
    payload = {
        "cycle": cycle,
        "fills": [fill.__dict__ for fill in fills],
        "timestamp": time.time(),
        "snapshot": {
            "total_equity": snapshot.total_equity,
            "strategies": {
                name: {
                    "cash": state.cash,
                    "positions": [pos.__dict__ for pos in state.positions],
                    "pnl": state.pnl.__dict__,
                }
                for name, state in snapshot.states.items()
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def run(args: argparse.Namespace) -> None:
    portfolio_config = load_portfolio_config(args.portfolio_config)
    strategy_params = {
        cfg.name: load_strategy_params(cfg.name, args.strategy_dir)
        for cfg in portfolio_config.strategies
    }

    venues = _determine_venues(strategy_params, args.default_timeframe)
    clients = _build_ccxt_clients(venues)

    broker = PaperBroker()
    executor = CCXTSignalExecutor(
        broker,
        fees_bps=args.fees_bps,
        slippage_bps=args.slip_bps,
        log=LOGGER,
    )

    orchestrator, portfolio, _, _, _ = build_multi_strategy_orchestrator(
        portfolio_config=portfolio_config,
        strategy_params=strategy_params,
        clients=clients,
        broker=broker,
        executor=executor,
        default_timeframe=args.default_timeframe,
        ohlcv_candles=args.ohlcv,
        log=LOGGER,
    )

    LOGGER.info(
        "Starting multi-strategy portfolio in %s mode with strategies: %s",
        args.mode,
        ", ".join(strategy_params.keys()),
    )

    artifacts_dir = Path(args.artifacts) if args.artifacts else None
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    cycle = 0

    try:
        while True:
            cycle += 1
            fills = orchestrator.run_cycle()
            snapshot = portfolio.snapshot()

            LOGGER.info(
                "Cycle %s produced %d fills; portfolio equity %.2f",
                cycle,
                len(fills),
                snapshot.total_equity,
            )

            if artifacts_dir:
                _dump_cycle_result(artifacts_dir / f"portfolio_cycle_{cycle}.json", cycle, fills, snapshot)

            if args.mode == "backtest" and cycle >= args.cycles:
                break

            if args.mode == "paper":
                time.sleep(args.interval)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:  # pragma: no cover - manual run convenience
        LOGGER.info("Received interrupt; shutting down portfolio runner.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--portfolio-config",
        default="config/portfolio.yaml",
        help="Path to the portfolio YAML configuration",
    )
    ap.add_argument(
        "--strategy-dir",
        default="config/strategy",
        help="Directory containing per-strategy configuration files",
    )
    ap.add_argument(
        "--mode",
        choices=["backtest", "paper"],
        default="backtest",
        help="Run a finite number of cycles (backtest) or loop forever (paper)",
    )
    ap.add_argument("--cycles", type=int, default=1, help="Number of cycles to run in backtest mode")
    ap.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Sleep interval between cycles when running in paper mode",
    )
    ap.add_argument(
        "--default-timeframe",
        default="1h",
        help="Default timeframe to request from CCXT when not specified in configs",
    )
    ap.add_argument("--ohlcv", type=int, default=30, help="Number of OHLCV candles to request per symbol")
    ap.add_argument("--fees-bps", type=float, default=10.0, help="Fee assumption in basis points")
    ap.add_argument(
        "--slip-bps",
        type=float,
        default=5.0,
        help="Slippage assumption in basis points applied to every fill",
    )
    ap.add_argument(
        "--artifacts",
        default="artifacts/portfolio",
        help="Directory for persisting cycle summaries (set empty to disable)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv or sys.argv[1:])
    run(args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
