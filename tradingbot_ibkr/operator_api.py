"""Authenticated local HTTP interface for OpenClaw and other operators."""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .operator_service import TradingOperatorService


def create_operator_app(
    service: TradingOperatorService,
    *,
    operator_token: str | None = None,
) -> FastAPI:
    """Create a narrow operator API.

    The API deliberately excludes arbitrary order submission, live-mode
    activation, position flattening and kill-switch reset endpoints.
    """

    expected_token = operator_token or os.getenv("TRADING_OPERATOR_TOKEN")
    if not expected_token:
        raise RuntimeError("TRADING_OPERATOR_TOKEN is required")

    app = FastAPI(title="Trading Bot Operator API", version="0.1.0")

    def authorize(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing operator token",
            )
        supplied = authorization[len(prefix) :]
        if not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid operator token",
            )

    Auth = Annotated[None, Depends(authorize)]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/operator/status")
    def operator_status(_: Auth) -> dict[str, object]:
        return service.snapshot()

    @app.get("/operator/orders")
    def orders(_: Auth) -> dict[str, object]:
        return {"orders": service.snapshot()["orders"]}

    @app.get("/operator/positions")
    def positions(_: Auth) -> dict[str, object]:
        return {"positions": service.snapshot()["positions"]}

    @app.post("/operator/start-paper")
    def start_paper(_: Auth) -> dict[str, object]:
        service.start()
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/pause")
    def pause(_: Auth) -> dict[str, object]:
        service.pause()
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/stop")
    def stop(_: Auth) -> dict[str, object]:
        service.stop(cancel_open_orders=True)
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/cancel-all")
    def cancel_all(_: Auth) -> dict[str, object]:
        cancelled = service.cancel_all_orders()
        return {"cancelled": len(cancelled), "status": service.snapshot()["status"]}

    @app.post("/operator/emergency-stop")
    def emergency_stop(_: Auth) -> dict[str, object]:
        service.latch_kill_switch()
        return {"status": service.snapshot()["status"]}

    return app


__all__ = ["create_operator_app"]
