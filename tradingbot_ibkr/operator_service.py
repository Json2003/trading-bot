"""Safe service boundary for dashboards and external operators such as OpenClaw."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Event, Lock, RLock, Thread, current_thread
from typing import Any

from tradingbot_core.risk import PaperRecoveryController, RecoveryCfg
from tradingbot_ibkr.execution.broker_base import Order, Position


@dataclass(frozen=True, slots=True)
class OperatorStatus:
    mode: str
    state: str
    engine_configured: bool
    engine_name: str | None
    kill_switch_latched: bool
    open_orders: int
    open_positions: int
    cycle_count: int
    last_cycle_at: str | None
    last_error: str | None
    recovery_state: str
    recovery_stable_cycles: int
    recovery_post_rearm_stable_cycles: int
    recovery_rearm_attempts: int
    recovery_can_auto_rearm: bool
    recovery_can_full_reset: bool


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
        engine_name: str | None = None,
        cycle_interval_seconds: float = 1.0,
        recovery_config: RecoveryCfg | None = None,
    ) -> None:
        if mode != "paper":
            raise ValueError("operator service rescue release supports paper mode only")
        if cycle_interval_seconds <= 0:
            raise ValueError("cycle_interval_seconds must be positive")

        self._broker = broker
        self._orchestrator = orchestrator
        self._mode = mode
        self._engine_name = engine_name or (
            type(orchestrator).__name__ if orchestrator is not None else None
        )
        self._cycle_interval_seconds = float(cycle_interval_seconds)
        self._state = "stopped"
        self._kill_switch_latched = False
        self._cycle_count = 0
        self._last_cycle_at: str | None = None
        self._last_error: str | None = None
        self._recovery = PaperRecoveryController(recovery_config)
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

        self._wait_for_cycle_thread(thread)
        if cancel_open_orders:
            self.cancel_all_orders()
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

    def latch_kill_switch(
        self,
        *,
        reason: str = "manual emergency stop",
        recoverable: bool = False,
    ) -> OperatorStatus:
        """Stop the paper service; manual emergency stops remain permanent.

        A trusted drawdown monitor may set ``recoverable=True``. It still must
        call :meth:`evaluate_recovery` repeatedly and pass every stability
        gate before the service can be re-armed.
        """

        with self._lock:
            self._kill_switch_latched = True
            self._state = "stopped"
            self._last_error = str(reason)
            self._recovery.trip(reason, manual=not recoverable, recoverable=recoverable)
            self._stop_event.set()
            thread = self._thread

        self._wait_for_cycle_thread(thread)
        self.cancel_all_orders()
        return self.status()

    def evaluate_recovery(
        self,
        *,
        current_drawdown_fraction: float,
        realized_volatility_fraction: float,
        engine_healthy: bool,
    ) -> OperatorStatus:
        """Feed one trusted paper-monitor observation into the recovery gate.

        This endpoint only evaluates observed state; it cannot force a reset.
        The broker must be flat and the engine healthy before the one-time
        drawdown re-arm is granted. Manual emergency stops and engine faults
        are not recoverable.
        """

        orders = self.open_orders()
        positions = self.positions()
        with self._lock:
            self._recovery.observe(
                current_drawdown_fraction=current_drawdown_fraction,
                realized_volatility_fraction=realized_volatility_fraction,
                engine_healthy=engine_healthy,
                open_orders=len(orders),
                open_positions=len(positions),
            )
            if self._kill_switch_latched and self._recovery.can_auto_rearm():
                self._recovery.auto_rearm()
                self._kill_switch_latched = False
                self._state = "stopped"
                self._last_error = "paper drawdown kill switch auto-rearmed after stability gate"
        return self.status()

    def complete_full_recovery_reset(self, *, human_approved: bool = False) -> OperatorStatus:
        """Clear recovery history after stability and explicit approval."""

        with self._lock:
            self._recovery.full_reset(human_approved=human_approved)
            self._last_error = None
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
            recovery = self._recovery.status()
            return OperatorStatus(
                mode=self._mode,
                state=self._state,
                engine_configured=self.engine_configured,
                engine_name=self._engine_name,
                kill_switch_latched=self._kill_switch_latched,
                open_orders=len(orders),
                open_positions=len(positions),
                cycle_count=self._cycle_count,
                last_cycle_at=self._last_cycle_at,
                last_error=self._last_error,
                recovery_state=recovery.state,
                recovery_stable_cycles=recovery.stable_cycles,
                recovery_post_rearm_stable_cycles=recovery.post_rearm_stable_cycles,
                recovery_rearm_attempts=recovery.rearm_attempts,
                recovery_can_auto_rearm=recovery.can_auto_rearm,
                recovery_can_full_reset=recovery.can_full_reset,
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
                    reason = str(
                        getattr(engine, "kill_switch_reason", "trading engine kill switch triggered")
                    )
                    recoverable = bool(getattr(engine, "kill_switch_recoverable", False))
                    self._fault(reason, recoverable=recoverable)
                    return
            except SystemExit as exc:
                self._fault("trading engine kill switch triggered", recoverable=False)
                raise RuntimeError("trading engine kill switch triggered") from exc
            except Exception as exc:
                message = f"trading engine fault: {type(exc).__name__}: {exc}"
                self._fault(message, recoverable=False)
                raise RuntimeError(message) from exc

            with self._lock:
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc).isoformat()

    def _wait_for_cycle_thread(self, thread: Thread | None) -> None:
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=max(2.0, self._cycle_interval_seconds * 2))
            if thread.is_alive():
                raise RuntimeError("trading engine did not stop within the safety timeout")
        with self._cycle_lock:
            pass

    def _fault(self, message: str, *, recoverable: bool = False) -> None:
        with self._lock:
            self._kill_switch_latched = True
            self._state = "faulted"
            self._last_error = message
            self._recovery.trip(message, recoverable=recoverable)
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
