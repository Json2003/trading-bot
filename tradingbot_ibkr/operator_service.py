"""Safe service boundary for dashboards and external operators such as OpenClaw."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Iterable

from tradingbot_ibkr.execution.broker_base import Order, Position


@dataclass(frozen=True, slots=True)
class OperatorStatus:
    mode: str
    state: str
    kill_switch_latched: bool
    open_orders: int
    open_positions: int


class TradingOperatorService:
    """Expose lifecycle and observation operations without arbitrary order entry.

    The service intentionally does not provide a method for submitting an ad-hoc
    order or resetting a latched kill switch. Those actions remain inside the
    deterministic trading engine and an authenticated manual recovery process.
    """

    def __init__(self, *, broker: Any, orchestrator: Any | None = None, mode: str = "paper") -> None:
        if mode != "paper":
            raise ValueError("operator service rescue release supports paper mode only")
        self._broker = broker
        self._orchestrator = orchestrator
        self._mode = mode
        self._state = "stopped"
        self._kill_switch_latched = False
        self._lock = RLock()

    def start(self) -> OperatorStatus:
        with self._lock:
            if self._kill_switch_latched:
                raise RuntimeError("kill switch is latched; manual recovery is required")
            self._state = "running"
            return self.status()

    def pause(self) -> OperatorStatus:
        with self._lock:
            if self._state == "running":
                self._state = "paused"
            return self.status()

    def stop(self, *, cancel_open_orders: bool = True) -> OperatorStatus:
        with self._lock:
            self._state = "stopped"
            if cancel_open_orders:
                self.cancel_all_orders()
            return self.status()

    def run_once(self) -> OperatorStatus:
        """Execute one deterministic strategy cycle while running."""

        with self._lock:
            if self._state != "running":
                raise RuntimeError("operator service must be running")
            if self._kill_switch_latched:
                raise RuntimeError("kill switch is latched")
            if self._orchestrator is None:
                raise RuntimeError("no orchestrator is configured")
            try:
                self._orchestrator.step()
            except SystemExit as exc:
                self._kill_switch_latched = True
                self._state = "stopped"
                self.cancel_all_orders()
                raise RuntimeError("trading engine kill switch triggered") from exc
            return self.status()

    def latch_kill_switch(self) -> OperatorStatus:
        """Emergency stop callable by trusted local monitoring."""

        with self._lock:
            self._kill_switch_latched = True
            self._state = "stopped"
            self.cancel_all_orders()
            return self.status()

    def cancel_all_orders(self) -> list[Any]:
        cancel_all = getattr(self._broker, "cancel_all_orders", None)
        if callable(cancel_all):
            return list(cancel_all())

        cancelled: list[Any] = []
        cancel_one = getattr(self._broker, "cancel_order", None)
        if not callable(cancel_one):
            raise RuntimeError("broker does not support order cancellation")
        for order in list(self.open_orders()):
            cancelled.append(cancel_one(order.id))
        return cancelled

    def open_orders(self) -> list[Order]:
        return list(self._broker.list_open_orders())

    def positions(self) -> list[Position]:
        return list(self._broker.list_positions())

    def status(self) -> OperatorStatus:
        return OperatorStatus(
            mode=self._mode,
            state=self._state,
            kill_switch_latched=self._kill_switch_latched,
            open_orders=len(self.open_orders()),
            open_positions=len(self.positions()),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": asdict(self.status()),
            "orders": [self._serialize(item) for item in self.open_orders()],
            "positions": [self._serialize(item) for item in self.positions()],
        }

    @staticmethod
    def _serialize(value: Any) -> dict[str, Any]:
        try:
            return asdict(value)
        except TypeError:
            return dict(vars(value))


__all__ = ["OperatorStatus", "TradingOperatorService"]
