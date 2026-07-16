"""Convenience wrapper exposing a ccxt client via the execution broker protocol."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, MutableMapping

from tradingbot_core.strategy import OrderIntent

from execution.adapters import CCXTBroker as AdapterBroker

from .broker_base import BrokerBase, Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class CCXTBroker(BrokerBase):
    """Translate :mod:`ccxt` order and position data to shared execution objects."""

    def __init__(
        self,
        exchange_id: str | None = None,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        secret: str | None = None,
        testnet: bool = False,
        account_type: str = "spot",
        log: logging.Logger | None = None,
    ) -> None:
        self._log = log or logger
        if client is None:
            if exchange_id is None:
                raise ValueError("exchange_id or client must be provided")
            try:  # pragma: no cover - optional dependency
                import ccxt  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ccxt is required to instantiate CCXTBroker") from exc

            exchange_cls = getattr(ccxt, exchange_id)
            params: dict[str, Any] = {"enableRateLimit": True}
            if api_key:
                params["apiKey"] = api_key
            if secret:
                params["secret"] = secret
            client = exchange_cls(params)
            if testnet and hasattr(client, "set_sandbox_mode"):
                client.set_sandbox_mode(True)

        self.client = client
        self._adapter = AdapterBroker(client=client, account_type=account_type, log=self._log)
        self._orders: MutableMapping[str, Order] = {}

    def _order_key(self, order: Order) -> str | None:
        for attr in ("client_order_id", "idemp_key", "id"):
            value = getattr(order, attr, None)
            if value:
                return str(value)
        metadata = order.metadata
        if isinstance(metadata, Mapping):
            for candidate in ("client_order_id", "idemp_key", "order_id"):
                value = metadata.get(candidate)
                if value:
                    return str(value)
        return None

    def _update_local_cache(self, orders: Iterable[Order]) -> None:
        for order in orders:
            key = self._order_key(order)
            if key:
                self._orders[key] = order

    def intent_to_order(self, intent: OrderIntent) -> Order:
        symbol = intent.symbol.split(":", 1)[-1]
        return Order.from_intent(intent, symbol=symbol)

    def submit_order(self, order: Order) -> OrderStatus:
        """Canonical broker entry point consumed by the reconciler."""

        return self.place(order)

    def place(self, order: Order) -> OrderStatus:
        client_id = self._order_key(order)
        if client_id and client_id in self._orders:
            cached = self._orders[client_id]
            raw_meta = cached.metadata if isinstance(cached.metadata, Mapping) else {}
            raw_payload = raw_meta.get("raw") if isinstance(raw_meta, Mapping) else None
            if isinstance(raw_payload, Mapping) and raw_payload:
                return OrderStatus(
                    client_id=client_id,
                    exchange_id=str(raw_payload.get("id") or cached.id),
                    status=str(raw_payload.get("status", cached.status)),
                    filled_qty=float(raw_payload.get("filled", cached.filled_quantity) or 0.0),
                    avg_price=(
                        float(raw_payload.get("average"))
                        if raw_payload.get("average") is not None
                        else cached.price
                    ),
                    raw=raw_payload,
                )

        order_type = (
            order.order_type or ("limit" if order.price is not None else "market")
        ).lower()
        price = None if order_type == "market" else order.price
        params: dict[str, Any] = {}
        if isinstance(order.metadata, Mapping):
            extra_params = order.metadata.get("params")
            if isinstance(extra_params, Mapping):
                params.update(extra_params)
        if client_id and "clientOrderId" not in params:
            params["clientOrderId"] = client_id

        placed = self._adapter.submit_order(
            order.symbol,
            order.side,
            order.quantity,
            order_type=order_type,
            price=price,
            client_id=client_id,
            params=params,
        )
        key = self._order_key(placed)
        if key:
            self._orders[key] = placed

        metadata = placed.metadata if isinstance(placed.metadata, Mapping) else {}
        raw_payload = metadata.get("raw", {}) if isinstance(metadata, Mapping) else {}
        if not isinstance(raw_payload, Mapping):
            raw_payload = {}
        filled = raw_payload.get("filled", placed.filled_quantity)
        avg_price = raw_payload.get("average", placed.price)
        return OrderStatus(
            client_id=(
                str(raw_payload.get("clientOrderId", client_id))
                if (client_id or raw_payload)
                else None
            ),
            exchange_id=str(raw_payload.get("id", placed.id)),
            status=str(raw_payload.get("status", placed.status)),
            filled_qty=float(filled or 0.0),
            avg_price=float(avg_price) if avg_price is not None else None,
            raw=raw_payload,
        )

    def list_open_orders(self) -> Iterable[Order]:
        orders = tuple(self._adapter.list_open_orders())
        self._update_local_cache(orders)
        return orders

    def list_positions(self) -> Iterable[Position]:
        return tuple(self._adapter.list_positions())

    def fetch_open_orders(self, symbol: str | None = None) -> Iterable[Order]:
        if symbol:
            try:
                payload = self.client.fetch_open_orders(symbol)
            except Exception as exc:  # pragma: no cover - network interaction
                self._log.warning("Failed to fetch open orders for %s via ccxt: %s", symbol, exc)
                return ()
            orders = [
                self._adapter._ingest_order(raw)  # type: ignore[attr-defined]
                for raw in payload
            ]
        else:
            orders = list(self.list_open_orders())
        self._update_local_cache(orders)
        return tuple(orders)

    def fetch_positions(self) -> Mapping[str, float]:
        positions = self._adapter.list_positions()
        return {pos.symbol: pos.quantity for pos in positions}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel one order by local id, idempotency key, or broker client id."""

        requested = str(order_id)
        if requested not in self._orders:
            self._update_local_cache(self.list_open_orders())
            for key, order in self._orders.items():
                if requested in {str(order.id), str(order.client_order_id), str(order.idemp_key)}:
                    requested = key
                    break
        return self.cancel(requested)

    def cancel(self, client_id: str) -> bool:
        order = self._orders.get(client_id)
        if not order:
            return False
        metadata = order.metadata if isinstance(order.metadata, Mapping) else {}
        raw_payload = metadata.get("raw", {}) if isinstance(metadata, Mapping) else {}
        exchange_id = (
            raw_payload.get("id") or raw_payload.get("orderId")
            if isinstance(raw_payload, Mapping)
            else None
        )
        exchange_id = exchange_id or order.id
        try:
            if not hasattr(self.client, "cancel_order"):
                return False
            self.client.cancel_order(exchange_id, order.symbol)
        except Exception as exc:  # pragma: no cover - network interaction
            self._log.warning("Failed to cancel order %s via ccxt: %s", client_id, exc)
            return False
        self._orders.pop(client_id, None)
        return True

    def cancel_all_orders(self) -> list[str]:
        """Cancel every order currently reported open by the exchange."""

        open_orders = tuple(self.list_open_orders())
        open_keys = [key for order in open_orders if (key := self._order_key(order))]
        cancelled: list[str] = []
        for key in open_keys:
            if self.cancel(key):
                cancelled.append(key)
        return cancelled


__all__ = ["CCXTBroker"]
