"""Adapter exposing a ccxt client via the :mod:`tradingbot_ibkr` broker protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import logging
import uuid

from tradingbot_ibkr.execution.broker_base import BrokerBase, Order, Position

logger = logging.getLogger(__name__)


@dataclass
class CCXTBroker(BrokerBase):
    """Translate ccxt order/position data into the shared broker protocol."""

    client: Any
    account_type: str = "spot"
    log: logging.Logger | None = None

    def __post_init__(self) -> None:
        self._log = self.log or logger
        self._orders: dict[str, Order] = {}

    def list_open_orders(self) -> Iterable[Order]:
        try:
            payload = self.client.fetch_open_orders()
        except Exception as exc:  # pragma: no cover - depends on network access
            self._log.warning("Failed to fetch open orders from ccxt: %s", exc)
            return [order for order in self._orders.values() if order.status == "open"]

        for raw in payload:
            self._ingest_order(raw)
        return [order for order in self._orders.values() if order.status == "open"]

    def list_positions(self) -> Iterable[Position]:
        positions: list[Position] = []
        if hasattr(self.client, "fetch_positions"):
            try:
                payload = self.client.fetch_positions()
            except Exception as exc:  # pragma: no cover - depends on network access
                self._log.warning("Failed to fetch positions via ccxt: %s", exc)
            else:
                positions.extend(self._transform_position(raw) for raw in payload if raw)
        else:
            try:
                balance: Mapping[str, Mapping[str, float]] = self.client.fetch_balance()
            except Exception as exc:  # pragma: no cover
                self._log.warning("Failed to fetch balance via ccxt: %s", exc)
            else:
                totals = balance.get("total", {})
                for symbol, quantity in totals.items():
                    qty = float(quantity)
                    if abs(qty) <= 0:
                        continue
                    positions.append(
                        Position(symbol=symbol, quantity=qty, average_price=None, metadata={"balance": quantity})
                    )
        return positions

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        order_type: str = "market",
        price: float | None = None,
        client_id: str | None = None,
        params: Mapping[str, object] | None = None,
    ) -> Order:
        client_order_id = client_id or uuid.uuid4().hex
        if client_order_id in self._orders:
            return self._orders[client_order_id]

        payload_params = dict(params or {})
        payload_params.setdefault("clientOrderId", client_order_id)
        if self.account_type:
            payload_params.setdefault("type", self.account_type)

        order = self.client.create_order(symbol, order_type, side, quantity, price, payload_params)
        broker_order = self._ingest_order(order, client_order_id=client_order_id)
        return broker_order

    # ------------------------------------------------------------------
    def _ingest_order(self, raw: Mapping[str, object], *, client_order_id: str | None = None) -> Order:
        cid = client_order_id or raw.get("clientOrderId") or raw.get("id") or uuid.uuid4().hex
        order_id = str(raw.get("id") or raw.get("orderId") or cid)
        filled = float(raw.get("filled") or raw.get("filledAmount") or raw.get("executed") or 0.0)
        status = str(raw.get("status") or ("closed" if filled else "open"))
        price = raw.get("average") or raw.get("price")
        broker_order = Order(
            id=order_id,
            symbol=str(raw.get("symbol")),
            side=str(raw.get("side")),
            quantity=float(raw.get("amount") or raw.get("quantity") or raw.get("origQty") or 0.0),
            filled_quantity=filled,
            status=status,
            price=float(price) if price else None,
            metadata={"raw": raw, "client_order_id": cid},
        )
        self._orders[cid] = broker_order
        return broker_order

    def _transform_position(self, raw: Mapping[str, object]) -> Position:
        qty = raw.get("contracts") or raw.get("positionAmt") or raw.get("size") or raw.get("amount") or 0.0
        price = raw.get("entryPrice") or raw.get("avgEntryPrice") or raw.get("markPrice")
        return Position(
            symbol=str(raw.get("symbol")),
            quantity=float(qty),
            average_price=float(price) if price else None,
            metadata={"raw": raw},
        )


__all__ = ["CCXTBroker"]
