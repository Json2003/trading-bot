from __future__ import annotations

from fastapi.testclient import TestClient

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService


def _client() -> TestClient:
    service = TradingOperatorService(broker=PaperBroker())
    app = create_operator_app(service, operator_token="test-token")
    return TestClient(app)


def test_status_requires_authentication() -> None:
    response = _client().get("/operator/status")

    assert response.status_code == 401


def test_invalid_token_is_rejected() -> None:
    response = _client().get(
        "/operator/status",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 403


def test_operator_can_start_and_emergency_stop_paper_service() -> None:
    client = _client()
    headers = {"Authorization": "Bearer test-token"}

    started = client.post("/operator/start-paper", headers=headers)
    stopped = client.post("/operator/emergency-stop", headers=headers)
    restarted = client.post("/operator/start-paper", headers=headers)

    assert started.status_code == 200
    assert started.json()["status"]["state"] == "running"
    assert stopped.status_code == 200
    assert stopped.json()["status"]["kill_switch_latched"] is True
    assert restarted.status_code == 500


def test_api_has_no_arbitrary_order_endpoint() -> None:
    client = _client()
    response = client.post(
        "/operator/order",
        headers={"Authorization": "Bearer test-token"},
        json={"symbol": "AAPL", "side": "buy", "quantity": 100},
    )

    assert response.status_code == 404
