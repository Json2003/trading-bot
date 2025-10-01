"""FastAPI application exposing order routing endpoints."""
from __future__ import annotations

from typing import NoReturn

from fastapi import FastAPI, HTTPException

from .models import OrderRequest
from .order_router import (
    OrderRouter,
    RouterNotConfiguredError,
    UnknownBrokerError,
)

app = FastAPI(title="Trading Bot Router API")
router = OrderRouter()


def _handle_router_errors(exc: RouterNotConfiguredError | UnknownBrokerError) -> NoReturn:
    if isinstance(exc, RouterNotConfiguredError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/orders")
def create_order(req: OrderRequest, account_id: str, broker: str | None = None):
    """Submit an order to the configured broker."""

    try:
        return router.place(account_id, req, broker_hint=broker)
    except (RouterNotConfiguredError, UnknownBrokerError) as exc:
        _handle_router_errors(exc)


@app.get("/positions")
def positions(account_id: str, broker: str | None = None):
    """Return open positions for ``account_id``."""

    try:
        return router.positions(account_id, broker_hint=broker)
    except (RouterNotConfiguredError, UnknownBrokerError) as exc:
        _handle_router_errors(exc)


@app.get("/accounts")
def accounts():
    """Return the configured brokers and their available accounts."""

    return router.accounts()


__all__ = ["app", "router"]
