"""Run the local paper-trading operator API.

The default runtime is a synthetic, credential-free smoke engine that exercises
the strategy, execution, broker, position and operator paths end to end. It is
not a live-market deployment and is not evidence of strategy profitability.
"""

from __future__ import annotations

import logging
import os

import uvicorn

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService
from tradingbot_ibkr.rescue_runtime import build_synthetic_paper_runtime

LOGGER = logging.getLogger("tradingbot.operator")


def build_service() -> TradingOperatorService:
    runtime_name = os.getenv("TRADING_OPERATOR_RUNTIME", "synthetic-smoke").strip().lower()
    cycle_seconds = float(os.getenv("TRADING_OPERATOR_CYCLE_SECONDS", "1.0"))

    if runtime_name in {"synthetic", "synthetic-smoke"}:
        runtime = build_synthetic_paper_runtime(
            steps=int(os.getenv("TRADING_OPERATOR_SYNTHETIC_STEPS", "2000")),
            seed=int(os.getenv("TRADING_OPERATOR_SYNTHETIC_SEED", "11")),
        )
        return TradingOperatorService(
            broker=runtime.broker,
            orchestrator=runtime.engine,
            mode="paper",
            engine_name=runtime.name,
            cycle_interval_seconds=cycle_seconds,
        )

    if runtime_name in {"none", "disabled"}:
        LOGGER.warning("Operator API started without a trading engine")
        return TradingOperatorService(
            broker=PaperBroker(),
            mode="paper",
            engine_name=None,
            cycle_interval_seconds=cycle_seconds,
        )

    raise RuntimeError(
        "unsupported TRADING_OPERATOR_RUNTIME; use 'synthetic-smoke' or 'none'"
    )


def build_app():
    return create_operator_app(build_service())


app = build_app()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.getenv("TRADING_OPERATOR_HOST", "127.0.0.1")
    port = int(os.getenv("TRADING_OPERATOR_PORT", "8765"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("operator API must bind to a loopback address")
    uvicorn.run(app, host=host, port=port, access_log=False)
