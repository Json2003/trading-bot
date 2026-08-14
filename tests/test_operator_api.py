from __future__ import annotations

from fastapi.testclient import TestClient

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService


class CountingEngine:
    def __init__(self) -> None:
        self.cycles = 0

    def step(self) -> None:
        self.cycles += 1


def _client(*, with_engine: bool = True) -> TestClient:
    engine = CountingEngine() if with_engine else None
    service = TradingOperatorService(
        broker=PaperBroker(),
        orchestrator=engine,
        cycle_interval_seconds=0.02,
    )
    app = create_operator_app(service, operator_token="test-token")
    return TestClient(app)


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_health_reports_engine_readiness_without_authentication() -> None:
    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["engine_configured"] is True
    assert response.headers["cache-control"] == "no-store"


def test_status_requires_authentication() -> None:
    with _client() as client:
        response = client.get("/operator/status")

    assert response.status_code == 401


def test_invalid_token_is_rejected() -> None:
    with _client() as client:
        response = client.get(
            "/operator/status",
            headers={"Authorization": "Bearer wrong-token"},
        )

    assert response.status_code == 403


def test_start_rejects_service_without_engine() -> None:
    with _client(with_engine=False) as client:
        response = client.post("/operator/start-paper", headers=_headers())

    assert response.status_code == 409
    assert "no trading engine" in response.json()["detail"]


def test_operator_can_start_and_emergency_stop_paper_service() -> None:
    with _client() as client:
        started = client.post("/operator/start-paper", headers=_headers())
        stopped = client.post("/operator/emergency-stop", headers=_headers())
        restarted = client.post("/operator/start-paper", headers=_headers())

    assert started.status_code == 200
    assert started.json()["status"]["state"] == "running"
    assert started.json()["status"]["engine_configured"] is True
    assert stopped.status_code == 200
    assert stopped.json()["status"]["kill_switch_latched"] is True
    assert restarted.status_code == 409
    assert "manual recovery" in restarted.json()["detail"]


def test_api_has_no_arbitrary_order_endpoint() -> None:
    with _client() as client:
        response = client.post(
            "/operator/order",
            headers=_headers(),
            json={"symbol": "AAPL", "side": "buy", "quantity": 100},
        )

    assert response.status_code == 404
