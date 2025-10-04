"""Simple in-memory broker implementation for demos and tests."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Dict, Mapping, Sequence

from .broker_base import Broker
from models import OrderRequest, OrderStatus, Position, Side


class PaperBroker(Broker):
    """A naive broker that immediately fills orders in-memory."""

    name = "paper"
    supports_crypto = True
    paper_trading = True

    def __init__(
        self,
        *,
        initial_positions: Mapping[str, Mapping[str, float]] | None = None,
        default_price: float = 0.0,
        default_cash: float = 0.0,
    ) -> None:
        """Create a new :class:`PaperBroker`.

        Args:
            initial_positions: Optional mapping of ``(account_id, symbol)`` pairs to
                starting quantities.  This is useful for seeding the broker with an
                existing book state when running tests.
            default_price: Fallback price used for fills when the request does not
                specify a limit or stop price.  Defaults to ``0.0``.
            default_cash: Default cash balance assigned to new accounts.
        """

        self._orders_by_broker_id: Dict[str, OrderStatus] = {}
        self._orders_by_client_id: Dict[str, str] = {}
        self._positions: Dict[tuple[str, str], Position] = {}
        self._order_accounts: Dict[str, str] = {}
        self._cash_balances: Dict[str, float] = {}
        self._default_price = float(default_price)
        self._default_cash = float(default_cash)

        for account_id, positions in (initial_positions or {}).items():
            normalized_account_id = str(account_id)
            for symbol, qty in positions.items():
                normalized_symbol = self.normalize_symbol(symbol)
                self._positions[(normalized_account_id, normalized_symbol)] = Position(
                    symbol=normalized_symbol,
                    qty=float(qty),
                    avg_price=self._default_price,
                    unrealized_pnl=0.0,
                    account_id=normalized_account_id,
                )
            self._cash_balances.setdefault(normalized_account_id, self._default_cash)

    # -- Broker interface --------------------------------------------------
    def place_order(self, account_id: str, req: OrderRequest) -> OrderStatus:
        account_id = str(account_id)
        client_order_id = req.client_order_id
        if client_order_id and client_order_id in self._orders_by_client_id:
            broker_order_id = self._orders_by_client_id[client_order_id]
            if self._order_accounts.get(broker_order_id) != account_id:
                raise KeyError(f"Order {broker_order_id} not found for account {account_id}")
            return self._orders_by_broker_id[broker_order_id]

        broker_order_id = uuid.uuid4().hex
        fill_price = (
            req.limit_price
            if req.limit_price is not None
            else req.stop_price
            if req.stop_price is not None
            else self._default_price
        )

        status = OrderStatus(
            broker=self.name,
            broker_order_id=broker_order_id,
            client_order_id=client_order_id,
            status="FILLED",
            filled_qty=req.qty,
            avg_price=fill_price,
            ts=time.time(),
            raw={"request": req.to_dict(include_none=True)},
        )

        self._orders_by_broker_id[broker_order_id] = status
        self._order_accounts[broker_order_id] = account_id
        if client_order_id:
            self._orders_by_client_id[client_order_id] = broker_order_id

        self._apply_fill(account_id, req, status)
        return status

    def cancel_order(self, account_id: str, broker_order_id: str) -> bool:
        account_id = str(account_id)
        status = self._orders_by_broker_id.get(broker_order_id)
        if status is None:
            return False
        if self._order_accounts.get(broker_order_id) != account_id:
            return False
        if status.status in {"FILLED", "CANCELLED"}:
            return False

        cancelled = replace(status, status="CANCELLED")
        self._orders_by_broker_id[broker_order_id] = cancelled
        self._order_accounts[broker_order_id] = account_id
        if cancelled.client_order_id:
            self._orders_by_client_id[cancelled.client_order_id] = broker_order_id
        return True

    def get_order(self, account_id: str, broker_order_id: str) -> OrderStatus:
        account_id = str(account_id)
        status = self._orders_by_broker_id[broker_order_id]
        if self._order_accounts.get(broker_order_id) != account_id:
            raise KeyError(f"Order {broker_order_id} not found for account {account_id}")
        return status

    def list_orders(self, account_id: str) -> Sequence[OrderStatus]:
        account_id = str(account_id)
        return [
            status
            for order_id, status in self._orders_by_broker_id.items()
            if self._order_accounts.get(order_id) == account_id
        ]

    def get_positions(self, account_id: str) -> Sequence[Position]:
        account_id = str(account_id)
        return [position for (acct, _), position in self._positions.items() if acct == account_id]

    def get_cash(self, account_id: str) -> float:
        account_id = str(account_id)
        return self._cash_balances.get(account_id, self._default_cash)

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    # -- Internal helpers --------------------------------------------------
    def _apply_fill(self, account_id: str, req: OrderRequest, status: OrderStatus) -> None:
        qty_delta = status.filled_qty
        if qty_delta == 0:
            return

        direction = 1 if req.side == Side.BUY else -1
        qty_delta *= direction
        symbol = self.normalize_symbol(req.symbol)
        key = (account_id, symbol)
        position = self._positions.get(key)

        current_qty = position.qty if position else 0.0
        new_qty = current_qty + qty_delta

        if abs(new_qty) < 1e-9:
            self._positions.pop(key, None)
        else:
            fill_price = status.avg_price if status.avg_price is not None else self._default_price
            if (
                position is None
                or current_qty == 0
                or (current_qty > 0 > new_qty)
                or (current_qty < 0 < new_qty)
            ):
                avg_price = fill_price
            elif direction > 0 and new_qty > 0:
                avg_price = self._weighted_average(
                    position.avg_price, current_qty, fill_price, qty_delta
                )
            elif direction < 0 and new_qty < 0:
                avg_price = self._weighted_average(
                    position.avg_price, current_qty, fill_price, qty_delta
                )
            else:
                avg_price = position.avg_price

            self._positions[key] = Position(
                symbol=symbol,
                qty=new_qty,
                avg_price=avg_price,
                unrealized_pnl=0.0,
                account_id=account_id,
            )

        notional = (status.avg_price or 0.0) * status.filled_qty
        balance = self._cash_balances.get(account_id, self._default_cash)
        if direction > 0:
            balance -= notional
        else:
            balance += notional
        self._cash_balances[account_id] = balance

    @staticmethod
    def _weighted_average(
        current_avg: float | None,
        current_qty: float,
        fill_price: float,
        fill_qty: float,
    ) -> float:
        if current_avg is None or current_qty == 0:
            return fill_price
        total_qty = current_qty + fill_qty
        if total_qty == 0:
            return fill_price
        return (current_avg * current_qty + fill_price * fill_qty) / total_qty


__all__ = ["PaperBroker"]
