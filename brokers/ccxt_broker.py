"""Lightweight CCXT broker adapter used by the reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    ccxt = None  # type: ignore

from tradingbot_core.strategy import OrderIntent

from .broker_base import Order, OrderStatus


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalise_status(raw_status: str | None) -> str:
    if not raw_status:
        return "open"
    status = raw_status.lower()
    mapping = {
        "open": "open",
        "new": "open",
        "pending": "open",
        "accepted": "open",
        "partially_filled": "partially_filled",
        "partial": "partially_filled",
        "closed": "filled",
        "filled": "filled",
        "canceled": "cancelled",
        "cancelled": "cancelled",
        "rejected": "rejected",
    }
    return mapping.get(status, status)


@dataclass(slots=True)
class CCXTBroker:
    """Expose a :mod:`ccxt` client via the minimal broker protocol used in tests."""

    exchange_id: str
    api_key: str | None = None
    secret: str | None = None
    testnet: bool = False
    client: Any | None = None
    _client_to_broker: MutableMapping[str, str] = field(init=False, repr=False)
    _client_symbols: MutableMapping[str, str | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            if ccxt is None:  # pragma: no cover - import guard
                raise ModuleNotFoundError("ccxt is required to use CCXTBroker")
            exchange_cls = getattr(ccxt, self.exchange_id)
            credentials = {
                "apiKey": self.api_key,
                "secret": self.secret,
                "enableRateLimit": True,
            }
            self.client = exchange_cls(credentials)
            if self.testnet and hasattr(self.client, "set_sandbox_mode"):
                self.client.set_sandbox_mode(True)
        self._client_to_broker = {}
        self._client_symbols = {}

    # ------------------------------------------------------------------
    def normalize_symbol(self, symbol: str) -> str:
        base = symbol.split(":", 1)[-1].strip()
        if "/" in base:
            return base.upper()
        if "-" in base:
            return base.replace("-", "/").upper()
        return base.upper()

    def intent_to_order(self, intent: OrderIntent) -> Order:
        metadata: Mapping[str, Any]
        raw_meta = intent.meta or {}
        metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {"meta": raw_meta}
        return Order(
            symbol=self.normalize_symbol(intent.symbol),
            quantity=float(intent.qty),
            side=str(intent.side).lower(),
            client_id=None,
            idemp_key=intent.idemp_key,
            price=intent.limit_price,
            order_type=str(intent.type).lower(),
            metadata=metadata,
        )

    def _status_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        idemp_key: Optional[str],
        fallback_client_id: Optional[str],
    ) -> OrderStatus:
        filled = _as_float(
            payload.get("filled")
            or payload.get("filledAmount")
            or payload.get("executed")
            or payload.get("amount_filled")
            or 0.0
        )
        avg_raw = (
            payload.get("average")
            or payload.get("avgPrice")
            or payload.get("avg_fill_price")
            or payload.get("price")
        )
        avg_price = None if avg_raw in (None, "") else _as_float(avg_raw)
        client_order_id = payload.get("clientOrderId") or fallback_client_id
        broker_order_id_raw = (
            payload.get("id")
            or payload.get("orderId")
            or payload.get("exchangeId")
            or payload.get("info", {}).get("orderId")
        )
        broker_order_id = str(broker_order_id_raw) if broker_order_id_raw else None
        status = _normalise_status(payload.get("status"))
        if not status and filled:
            status = "filled"
        elif not status:
            status = "open"
        result = OrderStatus(
            status=status,
            client_id=client_order_id,
            idemp_key=idemp_key or client_order_id,
            filled_quantity=filled,
            avg_price=avg_price,
            broker_order_id=broker_order_id,
            raw=payload,
        )
        if client_order_id:
            symbol = payload.get("symbol")
            if symbol:
                self._client_symbols[client_order_id] = symbol
            if broker_order_id:
                self._client_to_broker[client_order_id] = broker_order_id
        return result

    def place(self, order: Order) -> OrderStatus:
        if self.client is None:  # pragma: no cover - defensive
            raise RuntimeError("CCXT client is not configured")
        params: Dict[str, Any] = {}
        extra_params = order.metadata.get("params") if isinstance(order.metadata, Mapping) else None
        if isinstance(extra_params, Mapping):
            params.update(extra_params)
        client_ref = order.client_id or order.idemp_key
        create_order = getattr(self.client, "create_order", None)
        if client_ref and callable(create_order):
            try:
                varnames = create_order.__code__.co_varnames  # type: ignore[attr-defined]
            except AttributeError:
                varnames = ()
            if "clientOrderId" in varnames:
                params.setdefault("clientOrderId", client_ref)
        order_type = (order.order_type or ("market" if order.price is None else "limit")).lower()
        price = None if order_type == "market" else order.price
        payload = self.client.create_order(  # type: ignore[call-arg]
            order.symbol,
            order_type,
            order.side,
            order.quantity,
            price,
            params,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("ccxt create_order must return a mapping")
        return self._status_from_payload(payload, idemp_key=order.idemp_key, fallback_client_id=client_ref)

    def fetch_open_orders(self, symbol: str | None = None) -> Sequence[OrderStatus]:
        if self.client is None:  # pragma: no cover - defensive
            return []
        fetcher = getattr(self.client, "fetch_open_orders", None)
        if not callable(fetcher):
            return []
        payload = fetcher(symbol) if symbol else fetcher()
        orders: List[OrderStatus] = []
        for raw in payload or []:
            if not isinstance(raw, Mapping):
                continue
            orders.append(
                self._status_from_payload(raw, idemp_key=raw.get("clientOrderId"), fallback_client_id=raw.get("clientOrderId"))
            )
        return orders

    def fetch_positions(self) -> Sequence[Mapping[str, Any]]:
        if self.client is None:  # pragma: no cover - defensive
            return []
        fetch_positions = getattr(self.client, "fetch_positions", None)
        positions: List[Mapping[str, Any]] = []
        if callable(fetch_positions):
            try:
                payload = fetch_positions()
            except Exception:  # pragma: no cover - network dependent
                payload = None
            if payload:
                for raw in payload:
                    if not raw:
                        continue
                    positions.append(self._normalise_position(raw))
                return positions
        fetch_balance = getattr(self.client, "fetch_balance", None)
        if not callable(fetch_balance):
            return positions
        try:
            balance = fetch_balance()
        except Exception:  # pragma: no cover - network dependent
            return positions
        totals = balance.get("total", {}) if isinstance(balance, Mapping) else {}
        for symbol, quantity in totals.items():
            qty = _as_float(quantity)
            if abs(qty) <= 0:
                continue
            positions.append(
                {
                    "symbol": self.normalize_symbol(str(symbol)),
                    "quantity": qty,
                    "avg_price": None,
                    "raw": balance,
                }
            )
        return positions

    def _normalise_position(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = raw.get("symbol") or raw.get("info", {}).get("symbol")
        qty = (
            raw.get("contracts")
            or raw.get("positionAmt")
            or raw.get("size")
            or raw.get("amount")
            or raw.get("total")
            or 0.0
        )
        avg_price = raw.get("entryPrice") or raw.get("avgEntryPrice") or raw.get("markPrice")
        return {
            "symbol": self.normalize_symbol(str(symbol)) if symbol else None,
            "quantity": _as_float(qty),
            "avg_price": None if avg_price in (None, "") else _as_float(avg_price),
            "raw": raw,
        }

    def cancel(self, client_id: str, *, symbol: str | None = None) -> bool:
        if self.client is None:  # pragma: no cover - defensive
            return False
        cancel_order = getattr(self.client, "cancel_order", None)
        if not callable(cancel_order):
            return False
        params: Dict[str, Any] = {}
        try:
            varnames = cancel_order.__code__.co_varnames  # type: ignore[attr-defined]
        except AttributeError:
            varnames = ()
        if "clientOrderId" in varnames:
            params.setdefault("clientOrderId", client_id)
        broker_id = self._client_to_broker.get(client_id, client_id)
        resolved_symbol = symbol or self._client_symbols.get(client_id)
        try:
            cancel_order(broker_id, resolved_symbol, params)
        except Exception:  # pragma: no cover - depends on exchange
            return False
        return True


__all__ = ["CCXTBroker"]
