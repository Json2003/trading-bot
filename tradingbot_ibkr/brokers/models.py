"""Data structures shared by broker integrations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    """Enumerates the supported order sides."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Supported order types for broker requests."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Supported time in force policies."""

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass(slots=True)
class OrderRequest:
    """Represents a request to submit an order to a broker."""

    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None


class OrderState(str, Enum):
    """High level state machine for order execution."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(slots=True)
class OrderStatus:
    """Represents the most recent status of an order."""

    broker_order_id: str
    state: OrderState
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    message: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: Optional[str] = None


@dataclass(slots=True)
class Position:
    """Represents an open position at the broker."""

    symbol: str
    quantity: float
    avg_price: float
    market_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
