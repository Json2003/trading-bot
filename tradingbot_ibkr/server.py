"""FastAPI application exposing health and metadata endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import os

from fastapi import FastAPI

from tradingbot_core.config import load_config
from tradingbot_core.logging_setup import setup_logging

_DEFAULT_ENV = os.getenv("TRADINGBOT_ENV", "paper")
_DEFAULT_STRATEGY = os.getenv("TRADINGBOT_STRATEGY", "sample_meanrev")
_START_TIME = datetime.now(timezone.utc)


def _build_meta_payload(env_name: str, strategy_name: str) -> dict[str, object]:
    bundle = load_config(env_name, strategy_name)
    now = datetime.now(timezone.utc)
    return {
        "environment": bundle.env.get("name", env_name),
        "strategy": bundle.strategy.get("name", strategy_name),
        "fees": bundle.fees,
        "config": bundle.as_dict(),
        "started_at": _START_TIME.isoformat(),
        "uptime_seconds": (now - _START_TIME).total_seconds(),
    }


def create_app() -> FastAPI:
    setup_logging(name="tradingbot_ibkr.server")
    app = FastAPI(title="Trading Bot IBKR", version="0.1.0")

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/meta")
    def meta(env: str | None = None, strategy: str | None = None) -> dict[str, object]:
        env_name = env or _DEFAULT_ENV
        strategy_name = strategy or _DEFAULT_STRATEGY
        return _build_meta_payload(env_name, strategy_name)

    return app


app = create_app()

__all__ = ["app", "create_app"]
