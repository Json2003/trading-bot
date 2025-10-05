"""Core domain models used across the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isclose
from typing import Any, Dict, Mapping, MutableMapping, Optional

__all__ = [
    "Allocation",
    "OrderRequest",
    "OrderStatus",
    "PortfolioBook",
    "Position",
    "Side",
    "OrderType",
    "TimeInForce",
]


@dataclass(slots=True)
class Allocation:
    """Percentage allocations to individual strategies.

    The percentages must be expressed as decimals (e.g. ``0.25`` for 25%)
    and sum to ``1``.  The mapping is copied to ensure subsequent
    mutations by the caller do not affect the allocation object.
    """

    per_strategy_pct: Mapping[str, float]
    tolerance: float = field(default=1e-6, repr=False)

    def __post_init__(self) -> None:
        if not self.per_strategy_pct:
            raise ValueError("at least one strategy allocation is required")

        # Create an immutable copy for downstream consumers.
        self.per_strategy_pct = dict(self.per_strategy_pct)

        invalid = [name for name, pct in self.per_strategy_pct.items() if pct < 0]
        if invalid:
            raise ValueError(
                "allocations must be non-negative; invalid strategies: "
                + ", ".join(sorted(invalid))
            )

        total = sum(self.per_strategy_pct.values())
        if not isclose(total, 1.0, abs_tol=self.tolerance):
            raise ValueError(
                "strategy allocations must sum to 1.0; "
                f"received total {total:.6f}"
            )

    def to_dict(self) -> Dict[str, float]:
        """Return a shallow copy of the allocation percentages."""

        return dict(self.per_strategy_pct)


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


class PortfolioBook:
    """Track strategy-level allocations and aggregate portfolio equity."""

    def __init__(self, base_equity: float, alloc: Allocation):
        if base_equity <= 0:
            raise ValueError("base_equity must be positive")

        self.base_equity = float(base_equity)
        self.alloc = alloc
        self.strategy_equity: MutableMapping[str, float] = {
            name: self.base_equity * pct for name, pct in self.alloc.per_strategy_pct.items()
        }
        # Keep a running history of total equity snapshots.
        self._equity_curve = [self.total_equity]

    @property
    def equity_curve(self) -> list[float]:
        """Historical total equity snapshot after each update."""

        return list(self._equity_curve)

    @property
    def total_equity(self) -> float:
        """Return the sum of all strategy sub-accounts."""

        return float(sum(self.strategy_equity.values()))

    def credit_pnl(self, strategy: str, pnl: float) -> None:
        """Apply ``pnl`` to ``strategy`` and persist the updated total equity."""

        if strategy not in self.strategy_equity:
            raise KeyError(f"unknown strategy '{strategy}'")

        self.strategy_equity[strategy] += pnl
        self._equity_curve.append(self.total_equity)
