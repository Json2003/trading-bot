"""Authenticated local HTTP interface for OpenClaw and other operators."""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .operator_service import TradingOperatorService


def create_operator_app(
    service: TradingOperatorService,
    *,
    operator_token: str | None = None,
) -> FastAPI:
    """Create the narrow, paper-only local operator API.

    Arbitrary order submission, live activation, risk editing, position
    flattening and kill-switch reset are intentionally absent.
    """

    expected_token = operator_token or os.getenv("TRADING_OPERATOR_TOKEN")
    if not expected_token:
        raise RuntimeError("TRADING_OPERATOR_TOKEN is required")

    app = FastAPI(
        title="Trading Bot Operator API",
        version="0.2.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(RuntimeError)
    async def operator_conflict(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

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
    def health() -> dict[str, object]:
        service_status = service.status()
        return {
            "status": "ok",
            "mode": service_status.mode,
            "state": service_status.state,
            "engine_configured": service_status.engine_configured,
            "kill_switch_latched": service_status.kill_switch_latched,
        }

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

    @app.on_event("shutdown")
    def shutdown() -> None:
        service.close()

    return app


__all__ = ["create_operator_app"]
