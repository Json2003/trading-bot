"""Core domain models used across the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class Side(str, Enum):
    """Order side of a trade."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TimeInForce(str, Enum):
    """Duration policies supported by the broker."""

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


def _without_none(values: Dict[str, Any], *, include_none: bool) -> Dict[str, Any]:
    """Return ``values`` with ``None`` entries removed unless requested otherwise."""

    if include_none:
        return values

    return {key: value for key, value in values.items() if value is not None}


@dataclass(slots=True)
class OrderRequest:
    symbol: str  # normalized (e.g., "AAPL", "BTC/USD")
    side: Side
    qty: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None  # bracket/oco/trailing params, route hints, etc.

    def to_dict(self, *, include_none: bool = False) -> Dict[str, Any]:
        """Serialize the request into a broker-friendly dictionary."""

        return _without_none(
            {
                "symbol": self.symbol,
                "side": self.side.value,
                "qty": self.qty,
                "order_type": self.order_type.value,
                "limit_price": self.limit_price,
                "stop_price": self.stop_price,
                "tif": self.tif.value,
                "client_order_id": self.client_order_id,
                "meta": self.meta,
            },
            include_none=include_none,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrderRequest":
        """Create a request object from a plain dictionary payload."""

        return cls(
            symbol=str(payload["symbol"]),
            side=Side(payload["side"]),
            qty=float(payload["qty"]),
            order_type=OrderType(payload["order_type"]),
            limit_price=(
                None if payload.get("limit_price") is None else float(payload["limit_price"])
            ),
            stop_price=(
                None if payload.get("stop_price") is None else float(payload["stop_price"])
            ),
            tif=TimeInForce(payload.get("tif", TimeInForce.DAY.value)),
            client_order_id=payload.get("client_order_id"),
            meta=payload.get("meta"),
        )

    def copy_with(self, **changes: Any) -> "OrderRequest":
        """Return a copy with the supplied fields updated."""

        return replace(self, **changes)


@dataclass(slots=True)
class RiskCfg:
    """Risk guardrails used by the execution and monitoring layers."""

    per_trade_risk_pct: float  # 0.5–2.0
    max_daily_loss_pct: float  # e.g. 3.0
    kill_switch_drawdown_pct: float  # e.g. 8.0
    max_leverage: float  # cap at 5–10 for futures

    def __post_init__(self) -> None:  # pragma: no cover - defensive validation
        if self.per_trade_risk_pct <= 0:
            raise ValueError("per_trade_risk_pct must be positive")
        if self.max_daily_loss_pct <= 0:
            raise ValueError("max_daily_loss_pct must be positive")
        if self.kill_switch_drawdown_pct <= 0:
            raise ValueError("kill_switch_drawdown_pct must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")

    def to_dict(self) -> Dict[str, float]:
        """Serialize the configuration into a simple dictionary."""

        return {
            "per_trade_risk_pct": self.per_trade_risk_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "kill_switch_drawdown_pct": self.kill_switch_drawdown_pct,
            "max_leverage": self.max_leverage,
        }


@dataclass(slots=True)
class OrderStatus:
    broker: str
    broker_order_id: str
    client_order_id: Optional[str]
    status: str  # NEW, PARTIAL, FILLED, CANCELED, REJECTED
    filled_qty: float
    avg_price: Optional[float]
    ts: float  # epoch seconds
    raw: Dict[str, Any]  # the untouched broker payload for audit

    def to_dict(self, *, include_none: bool = False) -> Dict[str, Any]:
        """Serialize the order status into a plain dictionary."""

        return _without_none(
            {
                "broker": self.broker,
                "broker_order_id": self.broker_order_id,
                "client_order_id": self.client_order_id,
                "status": self.status,
                "filled_qty": self.filled_qty,
                "avg_price": self.avg_price,
                "ts": self.ts,
                "raw": self.raw,
            },
            include_none=include_none,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrderStatus":
        """Instantiate ``OrderStatus`` from a broker payload."""

        return cls(
            broker=str(payload["broker"]),
            broker_order_id=str(payload["broker_order_id"]),
            client_order_id=payload.get("client_order_id"),
            status=str(payload["status"]),
            filled_qty=float(payload["filled_qty"]),
            avg_price=(None if payload.get("avg_price") is None else float(payload["avg_price"])),
            ts=float(payload["ts"]),
            raw=dict(payload.get("raw", {})),
        )

    def copy_with(self, **changes: Any) -> "OrderStatus":
        """Return a modified copy of the current status object."""

        return replace(self, **changes)


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    unrealized_pnl: float
    account_id: str

    def to_dict(self, *, include_none: bool = False) -> Dict[str, Any]:
        """Serialize the position for storage or transport."""

        return _without_none(
            {
                "symbol": self.symbol,
                "qty": self.qty,
                "avg_price": self.avg_price,
                "unrealized_pnl": self.unrealized_pnl,
                "account_id": self.account_id,
            },
            include_none=include_none,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Position":
        """Create a position from a plain mapping."""

        return cls(
            symbol=str(payload["symbol"]),
            qty=float(payload["qty"]),
            avg_price=float(payload["avg_price"]),
            unrealized_pnl=float(payload["unrealized_pnl"]),
            account_id=str(payload["account_id"]),
        )

    def copy_with(self, **changes: Any) -> "Position":
        """Return a new ``Position`` with the supplied changes applied."""

        return replace(self, **changes)
