from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tradingbot_ibkr.execution.paper_broker import PaperBroker
from tradingbot_ibkr.operator_api import create_operator_app
from tradingbot_ibkr.operator_service import TradingOperatorService
from tradingbot_ibkr.tws_workstation import TwsAccountReader
from tradingbot_ibkr.workstation_simulator import NyseWorkstationSimulator

AUTH = {"Authorization": "Bearer workstation-test"}
PAPER = "DU1234567"
LIVE = "U7654321"


class FakeTws:
    def __init__(self, accounts=(PAPER, LIVE)):
        self.accounts = accounts
        self.client = SimpleNamespace(serverVersion=lambda: 178)
        self.events = []
        self.chart_error = False

    async def connectAsync(self, host, port, **options):
        self.events.append(("connect", host, port, options))

    def managedAccounts(self):
        self.events.append(("accounts",))
        return list(self.accounts)

    async def accountSummaryAsync(self, account):
        self.events.append(("balances", account))
        return [
            SimpleNamespace(account=account, tag="NetLiquidation", currency="USD", value="5124.50"),
            SimpleNamespace(account=account, tag="SettledCash", currency="CAD", value="1845.25"),
            SimpleNamespace(account="U0000111", tag="NetLiquidation", currency="USD", value="999999"),
            SimpleNamespace(account=account, tag="AvailableFunds", currency="USD", value="NaN"),
        ]

    def positions(self, account):
        contract = SimpleNamespace(localSymbol="IBM", symbol="IBM", secType="STK", currency="USD")
        return [
            SimpleNamespace(account=account, contract=contract, position=3, avgCost=190.75),
            SimpleNamespace(account="U0000111", contract=contract, position=99, avgCost=176),
        ]

    async def reqHistoricalDataAsync(self, contract, **options):
        self.events.append(("chart", contract, options))
        if self.chart_error:
            raise RuntimeError("data not enabled")
        dt = datetime(2026, 11, 20, 14, 30, tzinfo=timezone.utc)
        return [SimpleNamespace(date=dt, open=190.0, high=194.5, low=189.0, close=192.0, volume=150)]

    def disconnect(self):
        self.events.append(("disconnect",))


def _install_fake(monkeypatch, fake):
    module = ModuleType("ib_async")
    module.IB = lambda: fake
    module.StartupFetchNONE = 0
    module.Stock = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, "ib_async", module)
    monkeypatch.setattr(TwsAccountReader, "_sdk_version", staticmethod(lambda: "2.1.0"))


@pytest.mark.parametrize(("mode", "port", "account"), [("paper", 7497, PAPER), ("live", 7496, LIVE)])
def test_account_read_uses_distinct_profile_filters_and_closes(monkeypatch, mode, port, account):
    fake = FakeTws()
    _install_fake(monkeypatch, fake)
    reader = TwsAccountReader({"IBKR_TWS_PAPER_ACCOUNT": PAPER, "IBKR_TWS_LIVE_ACCOUNT": LIVE})
    result = asyncio.run(reader.check(mode, "IBM"))

    assert fake.events[0] == ("connect", "127.0.0.1", port, {
        "clientId": 71 if mode == "paper" else 72,
        "timeout": 4, "readonly": True, "raiseSyncErrors": True, "fetchFields": 0,
    })
    assert fake.events[2] == ("balances", account)
    assert fake.events[-1] == ("disconnect",)
    assert result["execution_enabled"] is False
    assert result["account"] == "••••" + account[-4:]
    assert account not in str(result)
    assert result["server_version"] == 178
    assert result["balances"] == {
        "equity": {"currency": "USD", "value": 5124.5},
        "settled_cash": {"currency": "CAD", "value": 1845.25},
    }
    assert len(result["positions"]) == 1
    assert result["positions"][0]["quantity"] == 3
    assert result["chart"]["bars"][0]["time"] == "2026-11-20T14:30:00+00:00"


def test_wrong_account_is_rejected_before_reading_account_data(monkeypatch):
    fake = FakeTws((LIVE,))
    _install_fake(monkeypatch, fake)
    reader = TwsAccountReader({"IBKR_TWS_PAPER_ACCOUNT": PAPER})
    with pytest.raises(RuntimeError, match="expected account"):
        asyncio.run(reader.check("paper", "IBM"))
    assert [event[0] for event in fake.events] == ["connect", "accounts", "disconnect"]


def test_chart_failure_does_not_claim_or_synthesize_market_bars(monkeypatch):
    fake = FakeTws()
    fake.chart_error = True
    _install_fake(monkeypatch, fake)
    reader = TwsAccountReader({"IBKR_TWS_LIVE_ACCOUNT": LIVE})
    result = asyncio.run(reader.check("live", "KO"))
    assert result["chart"]["bars"] == []
    assert "unavailable" in result["chart"]["message"]
    assert result["positions"][0]["symbol"] == "IBM"


def test_invalid_configuration_never_connects(monkeypatch):
    fake = FakeTws()
    _install_fake(monkeypatch, fake)
    reader = TwsAccountReader({"IBKR_TWS_PAPER_ACCOUNT": PAPER, "IBKR_TWS_LIVE_ACCOUNT": PAPER})
    with pytest.raises(ValueError, match="separate"):
        asyncio.run(reader.check("paper", "IBM"))
    assert fake.events == []
    with pytest.raises(RuntimeError, match="IBKR_TWS_PAPER_ACCOUNT"):
        asyncio.run(TwsAccountReader({}).check("paper", "IBM"))
    assert fake.events == []


def test_authenticated_reads_and_shared_web_assets(monkeypatch):
    fake = FakeTws()
    _install_fake(monkeypatch, fake)
    broker = PaperBroker()
    service = TradingOperatorService(broker=broker, orchestrator=NyseWorkstationSimulator(broker))
    with TestClient(create_operator_app(
        service, operator_token="workstation-test",
        tws_reader=TwsAccountReader({"IBKR_TWS_PAPER_ACCOUNT": PAPER}),
    )) as client:
        page = client.get("/workstation")
        assert page.status_code == 200 and "connect-src 'self'" in page.text
        assert "browser-bridge.js" in page.text and "workstation-test" not in page.text
        assert client.get("/workstation-assets/operator.css").status_code == 200
        assert client.get("/operator/tws-settings").status_code == 401
        assert client.get("/operator/tws/paper/IBM").status_code == 401
        assert client.get("/operator/tws/paper/BTC", headers=AUTH).status_code == 422
        assert client.post("/operator/tws/live/IBM", headers=AUTH).status_code == 405
        assert client.post("/operator/start-live", headers=AUTH).status_code == 404
        assert fake.events == []
        settings = client.get("/operator/tws-settings", headers=AUTH).json()
        assert settings["profiles"]["paper"]["configured"] is True
        assert settings["profiles"]["live"]["configured"] is False
        assert client.get("/operator/tws/paper/IBM", headers=AUTH).json()["execution_enabled"] is False
        sim = client.get("/operator/status", headers=AUTH).json()
        assert sim["status"]["mode"] == "paper"
        assert sim["workspace"]["source"] == "synthetic_nyse"
        assert sim["workspace"]["cash"] == 5000
        assert set(sim["workspace"]["charts"]) == {"IBM", "JPM", "KO"}


def test_sim_orders_fill_only_on_following_bar_and_charge_costs():
    broker = PaperBroker()
    engine = NyseWorkstationSimulator(broker)
    engine.step()
    engine.step()
    [order] = broker.list_open_orders()
    assert order.quantity == int(order.quantity)
    assert engine.workstation_snapshot()["cash"] == 5000
    assert not list(broker.list_positions())

    engine.step()
    [position] = broker.list_positions()
    expected_price = engine.workstation_snapshot()["charts"][order.symbol][-1]["open"]
    assert not list(broker.list_open_orders())
    assert position.average_price == expected_price
    assert engine.workstation_snapshot()["cash"] == pytest.approx(
        5000 - order.quantity * expected_price * 1.0043
    )
