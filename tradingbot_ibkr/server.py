"""FastAPI application exposing health and metadata endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping
import os

from fastapi import FastAPI

from tradingbot_core.config import ConfigBundle, load_config
from tradingbot_core.logging_setup import setup_logging
from tradingbot_ibkr.execution import BrokerBase, PaperBroker, Reconciler, RiskLimits
from src.settings import AppSettings

_START_TIME = datetime.now(timezone.utc)


def _risk_limits_from_bundle(bundle: ConfigBundle) -> RiskLimits | None:
    risk_cfg = bundle.env.get("risk") or {}
    required_keys = {"max_daily_loss_pct", "kill_switch_drawdown_pct", "max_position_risk_pct"}
    if not required_keys.issubset(risk_cfg):
        return None
    try:
        return RiskLimits(
            max_daily_loss_pct=float(risk_cfg["max_daily_loss_pct"]),
            kill_switch_drawdown_pct=float(risk_cfg["kill_switch_drawdown_pct"]),
            max_position_risk_pct=float(risk_cfg["max_position_risk_pct"]),
        )
    except (TypeError, ValueError):
        return None


def _build_meta_payload(
    env_name: str,
    strategy_name: str,
    *,
    settings: AppSettings | None = None,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    bundle = load_config(env_name, strategy_name)
    now = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "environment": bundle.env.get("name", env_name),
        "strategy": bundle.strategy.get("name", strategy_name),
        "fees": bundle.fees,
        "config": bundle.as_dict(),
        "started_at": _START_TIME.isoformat(),
        "uptime_seconds": (now - _START_TIME).total_seconds(),
        "version": "0.1.0",
    }

    if settings is not None:
        payload["runtime"] = {"mode": settings.TB_MODE, "broker": settings.BROKER}

    limits = _risk_limits_from_bundle(bundle)
    if limits:
        payload["risk_limits"] = limits.as_dict()

    if overrides:
        payload.update(overrides)

    return payload


def create_app(
    *,
    env_name: str | None = None,
    strategy_name: str | None = None,
    broker: BrokerBase | None = None,
    limits: RiskLimits | None = None,
) -> FastAPI:
    settings = AppSettings()
    logger = setup_logging(name="tradingbot_ibkr.server", level=settings.LOG_LEVEL)

    default_env = env_name or os.getenv("TRADINGBOT_ENV", settings.TB_MODE)
    default_strategy = strategy_name or os.getenv("TRADINGBOT_STRATEGY", "sample_meanrev")

    bundle = load_config(default_env, default_strategy)
    resolved_limits = limits or _risk_limits_from_bundle(bundle)

    broker_impl = broker or PaperBroker()
    reconciler = Reconciler(broker_impl, limits=resolved_limits, logger=logger)

    app = FastAPI(title="Trading Bot IBKR", version="0.1.0")
    app.state.settings = settings
    app.state.reconciler = reconciler
    app.state.broker = broker_impl
    app.state.default_env = default_env
    app.state.default_strategy = default_strategy
    app.state.risk_limits = resolved_limits

    @app.get("/health")
    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/meta")
    def meta(env: str | None = None, strategy: str | None = None) -> dict[str, object]:
        env_name = env or app.state.default_env
        strategy_name = strategy or app.state.default_strategy

        overrides: dict[str, object] = {}
        limits_for_payload = (
            app.state.risk_limits.as_dict() if app.state.risk_limits else None
        )
        if limits_for_payload:
            overrides.setdefault("risk_limits", limits_for_payload)

        return _build_meta_payload(
            env_name,
            strategy_name,
            settings=app.state.settings,
            overrides=overrides,
        )

    return app


app = create_app()

__all__ = ["app", "create_app"]
