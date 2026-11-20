"""Run the local simulator, read-only TWS views and research API.

The default trading runtime is a synthetic, credential-free NYSE demo. The
research lab uses local CSV datasets and writes job artifacts locally. Neither
component enables live trading or demonstrates profitability.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    os.environ["DISABLE_STUBS"] = "1"
    # The repository contains optional demo shims named numpy/pandas/requests.
    # Put installed packages first for the operator, including direct runs.
    repository = Path(__file__).resolve().parents[1]
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != repository]
    sys.path.append(str(repository))

import uvicorn

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService
from tradingbot_ibkr.paper_lab_automation import PaperLabAutomationService
from tradingbot_ibkr.rescue_runtime import build_synthetic_paper_runtime
from tradingbot_ibkr.strategy_candidates import StrategyCandidateRegistry
from tradingbot_ibkr.tws_workstation import TwsAccountReader
from tradingbot_ibkr.workstation_simulator import NyseWorkstationSimulator

LOGGER = logging.getLogger("tradingbot.operator")
ROOT = Path(__file__).resolve().parents[1]


def build_service() -> TradingOperatorService:
    runtime_name = os.getenv("TRADING_OPERATOR_RUNTIME", "nyse-demo").strip().lower()
    cycle_seconds = float(os.getenv("TRADING_OPERATOR_CYCLE_SECONDS", "1.0"))

    if runtime_name == "nyse-demo":
        broker = PaperBroker()
        return TradingOperatorService(
            broker=broker,
            orchestrator=NyseWorkstationSimulator(broker),
            mode="paper",
            engine_name="synthetic-nyse-demo",
            cycle_interval_seconds=cycle_seconds,
        )

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
        "unsupported TRADING_OPERATOR_RUNTIME; use 'nyse-demo', 'synthetic-smoke', or 'none'"
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
    artifact_root = Path(
        os.getenv("TRADING_RESEARCH_ARTIFACT_ROOT", str(ROOT / "var" / "paper_lab"))
    )
    return create_operator_app(
        build_service(),
        research_service=build_research_service(),
        candidate_registry=StrategyCandidateRegistry(artifact_root / "candidates.json"),
        tws_reader=TwsAccountReader(),
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
