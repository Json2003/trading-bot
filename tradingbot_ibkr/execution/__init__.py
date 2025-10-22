<<<<<<< HEAD
"""Execution utilities including smart order routing."""

from .smart_router import SmartOrderRouter  # noqa: F401

__all__ = ["SmartOrderRouter"]
=======
"""Execution layer primitives for IBKR and paper trading back-ends."""

from .broker_base import BrokerBase, Order, OrderStatus, Position
from .ccxt_broker import CCXTBroker
from .paper_broker import PaperBroker
from .reconciler import ReconciliationReport, Reconciler, RiskEvaluation, RiskLimits

__all__ = [
    "BrokerBase",
    "Order",
    "OrderStatus",
    "Position",
    "CCXTBroker",
    "PaperBroker",
    "ReconciliationReport",
    "Reconciler",
    "RiskEvaluation",
    "RiskLimits",
]
>>>>>>> origin/main
