"""Authenticated local HTTP interface for trading and research operators."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .operator_service import TradingOperatorService
from .paper_lab_automation import LabRunSpec, PaperLabAutomationService
from .strategy_candidates import StrategyCandidateRegistry
from .tws_workstation import TwsAccountReader

WORKSTATION_ASSETS = Path(__file__).resolve().parents[1] / "dashboard" / "electron-app" / "renderer"


class LabRunRequest(BaseModel):
    """Bounded research request; executable code and promotion are forbidden."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=240)
    generations: int = Field(default=3, ge=1, le=6)
    accounts_per_generation: int = Field(default=12, ge=4, le=24)
    final_holdout_fraction: float = Field(default=0.25, ge=0.20, le=0.40)
    seed: int = Field(default=7, ge=0, le=2_147_483_647)


class RecoveryCheckRequest(BaseModel):
    """Observed metrics used by the guarded paper recovery policy."""

    model_config = ConfigDict(extra="forbid")

    current_drawdown_fraction: float = Field(ge=0.0, le=1.0)
    realized_volatility_fraction: float = Field(ge=0.0, le=1.0)
    engine_healthy: bool = True


def create_operator_app(
    service: TradingOperatorService,
    *,
    operator_token: str | None = None,
    research_service: PaperLabAutomationService | None = None,
    candidate_registry: StrategyCandidateRegistry | None = None,
    tws_reader: TwsAccountReader | None = None,
) -> FastAPI:
    """Create the local paper-only operator and research API.

    Arbitrary order submission, live activation, risk editing, position
    flattening, automatic strategy promotion and forced kill-switch reset are
    absent. Recovery checks can only earn a bounded paper re-arm from trusted
    observed metrics; manual stops and engine faults remain latched.
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

    def authorize(request: Request) -> None:
        prefix = "Bearer "
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing operator token")
        supplied = authorization[len(prefix) :]
        if not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid operator token")

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
    def operator_status(_: None = Depends(authorize)) -> dict[str, object]:
        return service.snapshot()

    @app.get("/operator/tws-settings")
    def tws_settings(_: None = Depends(authorize)) -> dict[str, object]:
        if tws_reader is None:
            raise RuntimeError("TWS account reader is not configured")
        return tws_reader.settings()

    @app.get("/operator/tws/{mode}/{symbol}")
    async def tws_snapshot(
        mode: Literal["paper", "live"],
        symbol: Literal["IBM", "JPM", "KO"],
        _: None = Depends(authorize),
    ) -> dict[str, object]:
        if tws_reader is None:
            raise RuntimeError("TWS account reader is not configured")
        return await tws_reader.check(mode, symbol)

    @app.get("/workstation", response_class=HTMLResponse)
    def workstation() -> HTMLResponse:
        page = WORKSTATION_ASSETS / "operator.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="Workstation files are unavailable in this install")
        body = page.read_text(encoding="utf-8").replace("connect-src 'none'", "connect-src 'self'")
        body = body.replace('href="operator.css"', 'href="/workstation-assets/operator.css"')
        body = body.replace('src="operator.js"', 'src="/workstation-assets/operator.js"')
        browser_bridge = '<script src="/workstation-assets/browser-bridge.js" defer></script>'
        body = body.replace("</head>", f"{browser_bridge}\n</head>")
        return HTMLResponse(body)

    @app.get("/workstation-assets/{asset}")
    def workstation_asset(asset: Literal["operator.css", "operator.js", "browser-bridge.js"]) -> FileResponse:
        file = WORKSTATION_ASSETS / asset
        if not file.is_file():
            raise HTTPException(status_code=404, detail="Workstation asset is unavailable")
        return FileResponse(file, media_type="text/css" if asset.endswith(".css") else "text/javascript")

    @app.get("/operator/orders")
    def orders(_: None = Depends(authorize)) -> dict[str, object]:
        return {"orders": service.snapshot()["orders"]}

    @app.get("/operator/positions")
    def positions(_: None = Depends(authorize)) -> dict[str, object]:
        return {"positions": service.snapshot()["positions"]}

    @app.post("/operator/start-paper")
    def start_paper(_: None = Depends(authorize)) -> dict[str, object]:
        service.start()
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/pause")
    def pause(_: None = Depends(authorize)) -> dict[str, object]:
        service.pause()
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/stop")
    def stop(_: None = Depends(authorize)) -> dict[str, object]:
        service.stop(cancel_open_orders=True)
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/cancel-all")
    def cancel_all(_: None = Depends(authorize)) -> dict[str, object]:
        cancelled = service.cancel_all_orders()
        return {"cancelled": len(cancelled), "status": service.snapshot()["status"]}

    @app.post("/operator/emergency-stop")
    def emergency_stop(_: None = Depends(authorize)) -> dict[str, object]:
        service.latch_kill_switch()
        return {"status": service.snapshot()["status"]}

    @app.post("/operator/recovery-check")
    def recovery_check(
        request: RecoveryCheckRequest,
        _: None = Depends(authorize),
    ) -> dict[str, object]:
        service.evaluate_recovery(
            current_drawdown_fraction=request.current_drawdown_fraction,
            realized_volatility_fraction=request.realized_volatility_fraction,
            engine_healthy=request.engine_healthy,
        )
        return {"status": service.snapshot()["status"]}

    @app.get("/research/datasets")
    def research_datasets(_: None = Depends(authorize)) -> dict[str, object]:
        return {"datasets": require_research().datasets()}

    @app.get("/research/jobs")
    def research_jobs(_: None = Depends(authorize)) -> dict[str, object]:
        return {"jobs": require_research().jobs()}

    @app.get("/research/jobs/{job_id}")
    def research_job(job_id: str, _: None = Depends(authorize)) -> dict[str, object]:
        return {"job": require_research().job(job_id)}

    @app.post("/research/jobs", status_code=status.HTTP_202_ACCEPTED)
    def start_research(request: LabRunRequest, _: None = Depends(authorize)) -> dict[str, object]:
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
    def stage_research_candidate(job_id: str, account_id: str, _: None = Depends(authorize)) -> dict[str, object]:
        if candidate_registry is None:
            raise RuntimeError("candidate registry is not configured")
        candidate = require_research().stage_finalist(job_id, account_id, candidate_registry)
        return {"candidate": candidate}

    @app.post("/research/jobs/{job_id}/cancel")
    def cancel_research(job_id: str, _: None = Depends(authorize)) -> dict[str, object]:
        return {"job": require_research().cancel(job_id)}

    @app.on_event("shutdown")
    def shutdown() -> None:
        service.close()
        if research_service is not None:
            research_service.close()

    return app


__all__ = ["LabRunRequest", "RecoveryCheckRequest", "create_operator_app"]
