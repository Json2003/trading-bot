"""Alpaca broker implementation."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from .broker_base import Broker
from models import OrderRequest, OrderStatus, Position

try:  # pragma: no cover - optional dependency
    import alpaca_trade_api as tradeapi  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    tradeapi = None  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover - simple assignment
    _IMPORT_ERROR = None


class AlpacaBroker(Broker):
    """Concrete :class:`Broker` implementation backed by Alpaca's REST API."""

    name = "alpaca"
    supports_crypto = True
    paper_trading = True

    def __init__(self, key: str, secret: str, base_url: str):
        if tradeapi is None:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "alpaca_trade_api is required to use AlpacaBroker"
            ) from _IMPORT_ERROR

        self._key = key
        self._secret = secret
        self._base_url = base_url
        self.api = tradeapi.REST(key, secret, base_url)

    def connect(self) -> None:
        """Validate credentials by fetching the account."""

        self.api.get_account()

    def normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        if "-" in symbol and "/" not in symbol:
            return symbol.replace("-", "/")
        return symbol

    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        payload = {
            "symbol": self.normalize_symbol(req.symbol),
            "qty": req.qty,
            "side": req.side.value.lower(),
            "type": req.order_type.value.lower().replace("_", "-"),
            "time_in_force": req.tif.value.lower(),
            "limit_price": req.limit_price,
            "stop_price": req.stop_price,
            "client_order_id": req.client_order_id,
        }

        order = self.api.submit_order(
            **{key: value for key, value in payload.items() if value is not None}
        )
        return self._order_to_status(order, client_order_id=req.client_order_id)

    def cancel_order(self, account_id: str, broker_order_id: str) -> bool:
        try:
            self.api.cancel_order(broker_order_id)
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            api_error = getattr(tradeapi, "APIError", None)
            if api_error and isinstance(exc, api_error):
                if getattr(exc, "status_code", None) == 404:
                    return False
            raise

    def get_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        order = self.api.get_order(broker_order_id)
        return self._order_to_status(order)

    def list_orders(self, account_id: str) -> Sequence[OrderStatus]:  # type: ignore[override]
        orders = self.api.list_orders()
        return [self._order_to_status(order) for order in orders]

    def get_positions(self, account_id: str) -> Sequence[Position]:  # type: ignore[override]
        positions = self.api.list_positions()
        result = []
        for position in positions:
            result.append(
                Position(
                    symbol=self.normalize_symbol(position.symbol),
                    qty=float(position.qty),
                    avg_price=float(position.avg_entry_price),
                    unrealized_pnl=float(getattr(position, "unrealized_pl", 0.0)),
                    account_id=str(getattr(position, "account_id", account_id)),
                )
            )
        return result

    def get_cash(self, account_id: str) -> float:
        account = self.api.get_account()
        return float(account.cash)

    def stream_events(
        self,
        handler: Callable[[Dict[str, Any]], None],
        *,
        channels: Optional[Iterable[str]] = None,
    ) -> Optional[asyncio.Task[Any]]:
        if tradeapi is None:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "alpaca_trade_api is required to use AlpacaBroker"
            ) from _IMPORT_ERROR

        channels = list(channels or ("trade_updates", "account_updates"))

        stream = tradeapi.stream.Stream(
            key_id=self._key,
            secret_key=self._secret,
            base_url=self._base_url,
        )

        def _wrap(channel: str) -> Callable[[Dict[str, Any]], None]:
            def _handler(data: Dict[str, Any]) -> None:
                handler({"channel": channel, "data": data})

            return _handler

        for channel in channels:
            if channel == "trade_updates":
                stream.subscribe_trade_updates(_wrap(channel))
            elif channel == "account_updates":
                stream.subscribe_account_updates(_wrap(channel))
            else:
                stream.subscribe(channel, _wrap(channel))

        async def _runner() -> None:
            await stream._run_forever()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_runner())
            return None
        else:  # pragma: no cover - depends on caller context
            return loop.create_task(_runner())

    @staticmethod
    def _order_to_status(order: Any, *, client_order_id: Optional[str] = None) -> OrderStatus:
        avg_price = getattr(order, "filled_avg_price", None)
        if avg_price in (None, ""):
            normalized_avg_price: Optional[float] = None
        else:
            normalized_avg_price = float(avg_price)

        raw_payload = getattr(order, "_raw", None)
        if raw_payload is None:
            raw_payload = dict(getattr(order, "__dict__", {}))
        else:
            raw_payload = dict(raw_payload)

        return OrderStatus(
            broker="alpaca",
            broker_order_id=str(order.id),
            client_order_id=client_order_id or getattr(order, "client_order_id", None),
            status=str(getattr(order, "status", "")).upper(),
            filled_qty=float(getattr(order, "filled_qty", 0) or 0),
            avg_price=normalized_avg_price,
            ts=time.time(),
            raw=raw_payload,
        )
