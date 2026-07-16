"""Run the local paper-trading operator API.

This rescue entry point intentionally creates a paper broker only. A later
milestone will inject the repaired trading orchestrator and a broker paper
account after their integration tests pass.
"""

from __future__ import annotations

import os

import uvicorn

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService


def build_app():
    broker = PaperBroker()
    service = TradingOperatorService(broker=broker, mode="paper")
    return create_operator_app(service)


app = build_app()


if __name__ == "__main__":
    host = os.getenv("TRADING_OPERATOR_HOST", "127.0.0.1")
    port = int(os.getenv("TRADING_OPERATOR_PORT", "8765"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("rescue operator API must bind to a loopback address")
    uvicorn.run(app, host=host, port=port)
