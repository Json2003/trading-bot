from __future__ import annotations

import pytest

from tradingbot_core.config import ConfigBundle, load_config


def test_load_config_returns_expected_sections() -> None:
    bundle = load_config("paper", "sample_meanrev")

    assert isinstance(bundle, ConfigBundle)
    assert bundle.env["name"] == "paper"
    assert bundle.strategy["name"] == "sample_meanrev"
    assert bundle.fees["name"] == "binance_spot"
    assert "runtime" in bundle.as_dict()


def test_runtime_overrides_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TB_MODE", "paper")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("BROKER", "IBKR")
    monkeypatch.setenv("IBKR_BASE_URL", "https://localhost:5000")
    monkeypatch.setenv("IBKR_CLIENT_ID", "1111")
    monkeypatch.setenv("IBKR_ACCOUNT_ID", "DU1234567")
    monkeypatch.setenv("EXCHANGE_ID", "binance")
    monkeypatch.setenv("EXCHANGE_API_KEY", "replace_me")
    monkeypatch.setenv("EXCHANGE_SECRET", "replace_me")

    bundle = load_config("paper", "sample_meanrev")
    runtime = bundle.runtime

    assert runtime["mode"] == "paper"
    assert runtime["logging"]["level"] == "INFO"
    assert runtime["broker"]["name"] == "IBKR"
    assert runtime["broker"]["ibkr"]["base_url"] == "https://localhost:5000"
    assert runtime["broker"]["ibkr"]["client_id"] == 1111
    assert runtime["broker"]["ibkr"]["account_id"] == "DU1234567"
    assert runtime["exchange"]["id"] == "binance"
    assert runtime["exchange"]["api_key"] == "replace_me"
    assert runtime["exchange"]["secret"] == "replace_me"
