"""Broker routing utilities for placing orders with risk checks."""
from __future__ import annotations

from typing import Dict, Optional

from .models import OrderRequest, OrderStatus, OrderType
from .broker_base import Broker


class OrderRouter:
    """Simple order router that performs pre-trade validations."""

    def __init__(self, brokers: Dict[str, Broker], default_broker: str):
        if not brokers:
            raise ValueError("at least one broker must be provided")

        if default_broker not in brokers:
            raise ValueError("default broker must exist in the brokers mapping")

        self.brokers = brokers
        self.default = default_broker

    def place(
        self,
        account_id: str,
        req: OrderRequest,
        broker_hint: Optional[str] = None,
    ) -> OrderStatus:
        """Place an order with the appropriate broker after risk checks."""

        broker_name = broker_hint or self.default

        try:
            broker = self.brokers[broker_name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"unknown broker '{broker_name}'") from exc

        self._risk_guard(req, broker, account_id)
        return broker.place_order(account_id, req)

    def _risk_guard(self, req: OrderRequest, broker: Broker, account_id: str) -> None:
        """Perform basic validations before handing the order to the broker."""

        if not req.symbol:
            raise ValueError("symbol must be provided")

        if req.qty <= 0:
            raise ValueError("quantity must be a positive value")

        if req.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if req.limit_price is None:
                raise ValueError("limit orders require a limit_price")
            if req.limit_price <= 0:
                raise ValueError("limit_price must be positive")

        if req.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if req.stop_price is None:
                raise ValueError("stop-based orders require a stop_price")
            if req.stop_price <= 0:
                raise ValueError("stop_price must be positive")

        if req.order_type is OrderType.TRAILING_STOP and req.stop_price is not None:
            if req.stop_price <= 0:
                raise ValueError("trailing stop price offset must be positive")

        if hasattr(broker, "supports_crypto") and hasattr(req, "meta"):
            asset_class = (req.meta or {}).get("asset_class")
            if asset_class == "CRYPTO" and not getattr(broker, "supports_crypto", False):
                raise ValueError(f"broker '{broker.name}' does not support crypto trading")
            if asset_class == "EQUITY" and not getattr(broker, "supports_equities", False):
                raise ValueError(f"broker '{broker.name}' does not support equity trading")
            if asset_class == "OPTION" and not getattr(broker, "supports_options", False):
                raise ValueError(f"broker '{broker.name}' does not support options trading")
            if asset_class == "FUTURE" and not getattr(broker, "supports_futures", False):
                raise ValueError(f"broker '{broker.name}' does not support futures trading")

        # placeholder for per-account risk rules; no-op for now
        _ = account_id
