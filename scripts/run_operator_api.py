"""Run the local paper-trading operator and research API.

The default trading runtime is a synthetic, credential-free smoke engine. The
research lab uses local CSV datasets and writes job artifacts locally. Neither
component enables live trading or demonstrates profitability.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService
from tradingbot_ibkr.paper_lab_automation import PaperLabAutomationService
from tradingbot_ibkr.rescue_runtime import build_synthetic_paper_runtime

LOGGER = logging.getLogger("tradingbot.operator")
ROOT = Path(__file__).resolve().parents[1]


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


def build_research_service() -> PaperLabAutomationService:
    dataset_root = Path(
        os.getenv("TRADING_RESEARCH_DATASET_ROOT", str(ROOT / "backtest" / "sample_data"))
    )
    artifact_root = Path(
        os.getenv("TRADING_RESEARCH_ARTIFACT_ROOT", str(ROOT / "var" / "paper_lab"))
    )
    return PaperLabAutomationService(
        dataset_root=dataset_root,
        artifact_root=artifact_root,
        max_generations=int(os.getenv("TRADING_RESEARCH_MAX_GENERATIONS", "6")),
        max_accounts_per_generation=int(
            os.getenv("TRADING_RESEARCH_MAX_ACCOUNTS", "24")
        ),
    )


def build_app():
    return create_operator_app(
        build_service(),
        research_service=build_research_service(),
    )


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
