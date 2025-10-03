from __future__ import annotations

import pytest

prometheus = pytest.importorskip("prometheus_client")
CollectorRegistry = prometheus.CollectorRegistry

from tradingbot_core.monitoring import AlertConfig, MonitoringHub


class DummySession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def post(self, url: str, json: dict[str, object] | None = None, timeout: int | float | None = None) -> None:
        self.calls.append((url, json))


def test_monitoring_hub_records_metrics_and_alerts() -> None:
    registry = CollectorRegistry()
    session = DummySession()
    hub = MonitoringHub(
        alert_config=AlertConfig(discord_webhook="https://discord.test"),
        session=session,
        registry=registry,
    )

    hub.record_fill(symbol="BTC", quantity=0.5, price=20000.0)
    hub.record_error(error_type="engine", message="failure")
    hub.record_kill_switch(
        breached_limits={"max_daily_loss_pct": 5.0},
        daily_loss_pct=6.0,
        drawdown_pct=2.0,
        position_risk_pct=1.0,
    )

    fill_count = registry.get_sample_value("tradingbot_fills_total", labels={"symbol": "BTC"})
    assert fill_count == 1.0

    error_count = registry.get_sample_value("tradingbot_errors_total", labels={"type": "engine"})
    assert error_count == 1.0

    kill_switch_count = registry.get_sample_value("tradingbot_kill_switch_trips_total")
    assert kill_switch_count == 1.0

    assert any(call[0] == "https://discord.test" for call in session.calls)
