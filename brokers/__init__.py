"""Broker implementations and abstractions."""

from typing import Any

from .broker_base import Broker
from .alpaca_broker import AlpacaBroker

__all__ = ["Broker", "AlpacaBroker", "Reconciler", "RiskLimits"]


def __getattr__(name: str) -> Any:
    if name in {"Reconciler", "RiskLimits"}:
        from .reconciler import Reconciler, RiskLimits

        return {"Reconciler": Reconciler, "RiskLimits": RiskLimits}[name]
    raise AttributeError(f"module 'brokers' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"Reconciler", "RiskLimits"})
