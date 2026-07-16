"""Safe service boundary for dashboards and external operators such as OpenClaw."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Event, Lock, RLock, Thread, current_thread
from typing import Any

from tradingbot_ibkr.execution.broker_base import Order, Position


@dataclass(frozen=True, slots=True)
class OperatorStatus:
    mode: str
    state: str
    engine_configured: bool
    kill_switch_latched: bool
    open_orders: int
    open_positions: int
    cycle_count: int
    last_cycle_at: str | None
    last_error: str | None


class TradingOperatorService:
    """Expose lifecycle and observation operations without arbitrary order entry.

    A configured engine must provide either ``step()`` or ``run_cycle()``. The
    service owns the background cycle thread and latches the safety stop whenever
    the engine raises, exits, or reports its own kill switch.
    """

    def __init__(
        self,
        *,
        broker: Any,
        orchestrator: Any | None = None,
        mode: str = "paper",
        cycle_interval_seconds: float = 1.0,
    ) -> None:
        if mode != "paper":
            raise ValueError("operator service rescue release supports paper mode only")
        if cycle_interval_seconds <= 0:
            raise ValueError("cycle_interval_seconds must be positive")

        self._broker = broker
        self._orchestrator = orchestrator
        self._mode = mode
        self._cycle_interval_seconds = float(cycle_interval_seconds)
        self._state = "stopped"
        self._kill_switch_latched = False
        self._cycle_count = 0
        self._last_cycle_at: str | None = None
        self._last_error: str | None = None
        self._lock = RLock()
        self._cycle_lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None

    @property
    def engine_configured(self) -> bool:
        engine = self._orchestrator
        return callable(getattr(engine, "step", None)) or callable(
            getattr(engine, "run_cycle", None)
        )

    def start(self) -> OperatorStatus:
        with self._lock:
            if self._kill_switch_latched:
                raise RuntimeError("kill switch is latched; manual recovery is required")
            if not self.engine_configured:
                raise RuntimeError("no trading engine is configured")
            if self._thread is not None and self._thread.is_alive():
                self._state = "running"
                return self.status()

            self._stop_event.clear()
            self._last_error = None
            self._state = "running"
            thread = Thread(
                target=self._run_loop,
                name="trading-operator-cycle-loop",
                daemon=True,
            )
            self._thread = thread

        thread.start()
        return self.status()

    def pause(self) -> OperatorStatus:
        with self._lock:
            if self._state == "running":
                self._state = "paused"
            return self.status()

    def stop(self, *, cancel_open_orders: bool = True) -> OperatorStatus:
        with self._lock:
            self._state = "stopped"
            self._stop_event.set()
            thread = self._thread

        if cancel_open_orders:
            self.cancel_all_orders()
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=max(2.0, self._cycle_interval_seconds * 2))
        return self.status()

    def close(self) -> None:
        """Stop the engine loop and cancel any remaining open orders."""

        self.stop(cancel_open_orders=True)

    def run_once(self) -> OperatorStatus:
        """Execute one deterministic strategy cycle while running."""

        with self._lock:
            if self._state != "running":
                raise RuntimeError("operator service must be running")
            if self._kill_switch_latched:
                raise RuntimeError("kill switch is latched")
            if not self.engine_configured:
                raise RuntimeError("no trading engine is configured")

        self._execute_cycle()
        return self.status()

    def latch_kill_switch(self) -> OperatorStatus:
        """Emergency stop callable by trusted local monitoring."""

        with self._lock:
            self._kill_switch_latched = True
            self._state = "stopped"
            self._last_error = "manual emergency stop"
            self._stop_event.set()
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
        orders = self.open_orders()
        positions = self.positions()
        with self._lock:
            return OperatorStatus(
                mode=self._mode,
                state=self._state,
                engine_configured=self.engine_configured,
                kill_switch_latched=self._kill_switch_latched,
                open_orders=len(orders),
                open_positions=len(positions),
                cycle_count=self._cycle_count,
                last_cycle_at=self._last_cycle_at,
                last_error=self._last_error,
            )

    def snapshot(self) -> dict[str, Any]:
        orders = self.open_orders()
        positions = self.positions()
        status_snapshot = asdict(self.status())
        status_snapshot["open_orders"] = len(orders)
        status_snapshot["open_positions"] = len(positions)
        return {
            "status": status_snapshot,
            "orders": [self._serialize(item) for item in orders],
            "positions": [self._serialize(item) for item in positions],
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                state = self._state
            if state == "paused":
                self._stop_event.wait(min(0.25, self._cycle_interval_seconds))
                continue
            if state != "running":
                break

            try:
                self._execute_cycle()
            except RuntimeError:
                break
            self._stop_event.wait(self._cycle_interval_seconds)

    def _execute_cycle(self) -> None:
        engine = self._orchestrator
        cycle = getattr(engine, "step", None)
        if not callable(cycle):
            cycle = getattr(engine, "run_cycle", None)
        if not callable(cycle):
            raise RuntimeError("no trading engine is configured")

        with self._cycle_lock:
            try:
                cycle()
                if bool(getattr(engine, "kill_switch_triggered", False)):
                    raise SystemExit("trading engine kill switch triggered")
            except SystemExit as exc:
                self._fault("trading engine kill switch triggered")
                raise RuntimeError("trading engine kill switch triggered") from exc
            except Exception as exc:
                message = f"trading engine fault: {type(exc).__name__}: {exc}"
                self._fault(message)
                raise RuntimeError(message) from exc

            with self._lock:
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc).isoformat()

    def _fault(self, message: str) -> None:
        with self._lock:
            self._kill_switch_latched = True
            self._state = "faulted"
            self._last_error = message
            self._stop_event.set()
        try:
            self.cancel_all_orders()
        except Exception:
            # The original engine failure remains the primary incident. Broker
            # reconciliation will surface any cancellation failure separately.
            pass

    @staticmethod
    def _serialize(value: Any) -> dict[str, Any]:
        try:
            return asdict(value)
        except TypeError:
            return dict(vars(value))


__all__ = ["OperatorStatus", "TradingOperatorService"]
