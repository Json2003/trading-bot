"""Monitoring utilities for recording metrics and dispatching alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping
import logging

import requests

try:  # pragma: no cover - optional dependency guard
    from prometheus_client import CollectorRegistry, Counter
except ModuleNotFoundError:  # pragma: no cover - executed when prometheus_client missing

    class CollectorRegistry:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            pass

    class _NoopCounter:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def labels(self, **_: object) -> "_NoopCounter":
            return self

        def inc(self, *_: object, **__: object) -> None:
            pass

    Counter = _NoopCounter  # type: ignore[assignment]


@dataclass(slots=True)
class AlertConfig:
    """Configuration for Discord and Telegram alert sinks."""

    discord_webhook: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


class MonitoringHub:
    """Aggregates Prometheus metrics and pushes alerts to chat systems."""

    def __init__(
        self,
        *,
        alert_config: AlertConfig | None = None,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._alert_config = alert_config or AlertConfig()
        self._session = session or requests.Session()
        self._logger = logger or logging.getLogger(__name__)
        self._registry = registry or CollectorRegistry()

        self._fills = Counter(
            "tradingbot_fills_total",
            "Number of fills observed",
            ["symbol"],
            registry=self._registry,
        )
        self._fill_notional = Counter(
            "tradingbot_fill_notional_total",
            "Total traded notional across fills",
            ["symbol"],
            registry=self._registry,
        )
        self._errors = Counter(
            "tradingbot_errors_total",
            "Number of errors emitted",
            ["type"],
            registry=self._registry,
        )
        self._kill_switches = Counter(
            "tradingbot_kill_switch_trips_total",
            "Kill-switch activations",
            registry=self._registry,
        )

    def record_fill(self, *, symbol: str, quantity: float, price: float) -> None:
        self._fills.labels(symbol=symbol).inc()
        self._fill_notional.labels(symbol=symbol).inc(quantity * price)
        self._emit_alert(f"Fill executed for {symbol}: {quantity} @ {price}")

    def record_error(self, *, error_type: str, message: str) -> None:
        self._errors.labels(type=error_type).inc()
        self._emit_alert(f"Error ({error_type}): {message}")

    def record_kill_switch(
        self,
        *,
        breached_limits: Mapping[str, float] | None = None,
        daily_loss_pct: float,
        drawdown_pct: float,
        position_risk_pct: float,
    ) -> None:
        self._kill_switches.inc()
        details: MutableMapping[str, float] = {
            "daily_loss_pct": daily_loss_pct,
            "drawdown_pct": drawdown_pct,
            "position_risk_pct": position_risk_pct,
        }
        if breached_limits:
            details.update(breached_limits)
        self._emit_alert(
            "Kill-switch triggered: "
            + ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
        )

    def _emit_alert(self, message: str) -> None:
        config = self._alert_config
        if not (config.discord_webhook or (config.telegram_bot_token and config.telegram_chat_id)):
            return

        if config.discord_webhook:
            try:
                self._session.post(config.discord_webhook, json={"content": message}, timeout=5)
            except Exception:
                self._logger.exception("Failed to dispatch Discord alert")
        if config.telegram_bot_token and config.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
                payload = {"chat_id": config.telegram_chat_id, "text": message}
                self._session.post(url, json=payload, timeout=5)
            except Exception:
                self._logger.exception("Failed to dispatch Telegram alert")


__all__ = ["AlertConfig", "MonitoringHub"]
