"""Authenticated local HTTP interface for trading and research operators."""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .operator_service import TradingOperatorService
from .strategy_candidates import StrategyCandidateRegistry
from .paper_lab_automation import LabRunSpec, PaperLabAutomationService


class LabRunRequest(BaseModel):
    """Bounded research request; executable code and promotion are forbidden."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=240)
    generations: int = Field(default=3, ge=1, le=6)
    accounts_per_generation: int = Field(default=12, ge=4, le=24)
    final_holdout_fraction: float = Field(default=0.25, ge=0.20, le=0.40)
    seed: int = Field(default=7, ge=0, le=2_147_483_647)


def create_operator_app(
    service: TradingOperatorService,
    *,
    operator_token: str | None = None,
    research_service: PaperLabAutomationService | None = None,
    candidate_registry: StrategyCandidateRegistry | None = None,
) -> FastAPI:
    """Create the local paper-only operator and research API.

    Arbitrary order submission, live activation, risk editing, position
    flattening, automatic strategy promotion and kill-switch reset are absent.
    """

    expected_token = operator_token or os.getenv("TRADING_OPERATOR_TOKEN")
    if not expected_token:
        raise RuntimeError("TRADING_OPERATOR_TOKEN is required")

    app = FastAPI(
        title="Trading Bot Operator API",
        version="0.3.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def secure_response(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(RuntimeError)
    async def conflict(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"resource not found: {exc.args[0]}"},
        )

    @app.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    def authorize(authorization: str | None = Header(default=None, alias="Authorization")) -> None:
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing operator token")
        supplied = authorization[len(prefix) :]
        if not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid operator token")

    Auth = Depends(authorize)

    def require_research() -> PaperLabAutomationService:
        if research_service is None:
            raise RuntimeError("paper research lab is not configured")
        return research_service

    @app.get("/health")
    def health() -> dict[str, object]:
        snapshot = service.status()
        return {
            "status": "ok",
            "mode": snapshot.mode,
            "state": snapshot.state,
            "engine_configured": snapshot.engine_configured,
            "kill_switch_latched": snapshot.kill_switch_latched,
            "paper_lab_configured": research_service is not None,
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

    @app.get("/research/datasets")
    def research_datasets(_: Auth) -> dict[str, object]:
        return {"datasets": require_research().datasets()}

    @app.get("/research/jobs")
    def research_jobs(_: Auth) -> dict[str, object]:
        return {"jobs": require_research().jobs()}

    @app.get("/research/jobs/{job_id}")
    def research_job(job_id: str, _: Auth) -> dict[str, object]:
        return {"job": require_research().job(job_id)}

    @app.post("/research/jobs", status_code=status.HTTP_202_ACCEPTED)
    def start_research(request: LabRunRequest, _: Auth) -> dict[str, object]:
        job = require_research().start(
            LabRunSpec(
                dataset_id=request.dataset_id,
                generations=request.generations,
                accounts_per_generation=request.accounts_per_generation,
                final_holdout_fraction=request.final_holdout_fraction,
                seed=request.seed,
            )
        )
        return {"job": job}

    @app.post("/research/jobs/{job_id}/stage/{account_id}")
    def stage_research_candidate(job_id: str, account_id: str, _: Auth) -> dict[str, object]:
        if candidate_registry is None:
            raise RuntimeError("candidate registry is not configured")
        candidate = require_research().stage_finalist(job_id, account_id, candidate_registry)
        return {"candidate": candidate}

    @app.post("/research/jobs/{job_id}/cancel")
    def cancel_research(job_id: str, _: Auth) -> dict[str, object]:
        return {"job": require_research().cancel(job_id)}

    @app.on_event("shutdown")
    def shutdown() -> None:
        service.close()
        if research_service is not None:
            research_service.close()

    return app


__all__ = ["LabRunRequest", "create_operator_app"]
