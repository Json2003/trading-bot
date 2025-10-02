"""Utilities to compare local state with broker state and enforce risk limits."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence
import logging

from .broker_base import BrokerBase, Order, Position


@dataclass(frozen=True)
class RiskLimits:
    """Container describing the risk thresholds enforced by the reconciler."""

    max_daily_loss_pct: float
    kill_switch_drawdown_pct: float
    max_position_risk_pct: float

    def __post_init__(self) -> None:  # pragma: no cover - small validation helper
        for name, value in (
            ("max_daily_loss_pct", self.max_daily_loss_pct),
            ("kill_switch_drawdown_pct", self.kill_switch_drawdown_pct),
            ("max_position_risk_pct", self.max_position_risk_pct),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "kill_switch_drawdown_pct": self.kill_switch_drawdown_pct,
            "max_position_risk_pct": self.max_position_risk_pct,
        }


@dataclass(frozen=True)
class RiskEvaluation:
    """Outcome of assessing account state against configured risk limits."""

    daily_loss_pct: float
    drawdown_pct: float
    position_risk_pct: float
    breached_limits: Sequence[str] = ()

    @property
    def kill_switch_triggered(self) -> bool:
        return any(
            limit in {"max_daily_loss_pct", "kill_switch_drawdown_pct"}
            for limit in self.breached_limits
        )


@dataclass(frozen=True)
class ReconciliationReport:
    missing_orders: tuple[str, ...] = ()
    unexpected_orders: tuple[Order, ...] = ()
    quantity_mismatches: Mapping[str, float] = field(default_factory=dict)
    position_deltas: Mapping[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_clean(self) -> bool:
        return not (self.missing_orders or self.unexpected_orders or self.quantity_mismatches or self.position_deltas)


class Reconciler:
    def __init__(
        self,
        broker: BrokerBase,
        *,
        quantity_tolerance: float = 1e-6,
        limits: RiskLimits | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._broker = broker
        self._quantity_tolerance = quantity_tolerance
        self._limits = limits
        self._logger = logger or logging.getLogger(__name__)

    def _coerce_orders(self, orders: Iterable[Order] | Mapping[str, Order]) -> MutableMapping[str, Order]:
        if isinstance(orders, Mapping):
            return dict(orders)
        return {order.id: order for order in orders}

    def _coerce_positions(self, positions: Mapping[str, float] | Iterable[Position]) -> MutableMapping[str, float]:
        if isinstance(positions, Mapping):
            return dict(positions)
        return {pos.symbol: pos.quantity for pos in positions}

    def reconcile(
        self,
        *,
        local_orders: Iterable[Order] | Mapping[str, Order],
        local_positions: Mapping[str, float] | Iterable[Position],
    ) -> ReconciliationReport:
        broker_orders = {order.id: order for order in self._broker.list_open_orders()}
        broker_positions = {position.symbol: position.quantity for position in self._broker.list_positions()}

        local_orders_map = self._coerce_orders(local_orders)
        local_positions_map = self._coerce_positions(local_positions)

        missing_orders = tuple(sorted(order_id for order_id in local_orders_map.keys() - broker_orders.keys()))
        unexpected_orders = tuple(broker_orders[oid] for oid in broker_orders.keys() - local_orders_map.keys())

        quantity_mismatches: dict[str, float] = {}
        for order_id in broker_orders.keys() & local_orders_map.keys():
            broker_order = broker_orders[order_id]
            local_order = local_orders_map[order_id]
            delta = broker_order.remaining() - local_order.remaining()
            if abs(delta) > self._quantity_tolerance:
                quantity_mismatches[order_id] = delta

        position_deltas: dict[str, float] = {}
        symbols = broker_positions.keys() | local_positions_map.keys()
        for symbol in symbols:
            broker_qty = broker_positions.get(symbol, 0.0)
            local_qty = local_positions_map.get(symbol, 0.0)
            delta = broker_qty - local_qty
            if abs(delta) > self._quantity_tolerance:
                position_deltas[symbol] = delta

        return ReconciliationReport(
            missing_orders=missing_orders,
            unexpected_orders=unexpected_orders,
            quantity_mismatches=quantity_mismatches,
            position_deltas=position_deltas,
        )

    def evaluate_risk(
        self,
        *,
        daily_loss_pct: float,
        drawdown_pct: float,
        position_risk_pct: float,
    ) -> RiskEvaluation:
        """Compare metrics against configured limits and log any breaches."""

        limits = self._limits
        breached: list[str] = []

        if limits:
            if daily_loss_pct > limits.max_daily_loss_pct:
                breached.append("max_daily_loss_pct")
            if drawdown_pct > limits.kill_switch_drawdown_pct:
                breached.append("kill_switch_drawdown_pct")
            if position_risk_pct > limits.max_position_risk_pct:
                breached.append("max_position_risk_pct")

            if breached:
                self._logger.warning(
                    "Risk limits breached",
                    extra={
                        "breached_limits": tuple(breached),
                        "daily_loss_pct": daily_loss_pct,
                        "drawdown_pct": drawdown_pct,
                        "position_risk_pct": position_risk_pct,
                    },
                )

        return RiskEvaluation(
            daily_loss_pct=daily_loss_pct,
            drawdown_pct=drawdown_pct,
            position_risk_pct=position_risk_pct,
            breached_limits=tuple(breached),
        )


__all__ = ["RiskLimits", "RiskEvaluation", "ReconciliationReport", "Reconciler"]
