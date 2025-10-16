from __future__ import annotations

import math
import pathlib

import pytest

pytest.importorskip("yaml")

from tradingbot_core.config import ConfigBundle, load_config


def test_load_config_returns_expected_sections() -> None:
    bundle = load_config("paper", "sample_meanrev")

    assert isinstance(bundle, ConfigBundle)
    assert bundle.env["name"] == "paper"
    assert bundle.strategy["name"] == "sample_meanrev"
    assert bundle.fees["name"] == "binance_spot"
    assert "runtime" in bundle.as_dict()


def test_fee_config_accepts_basis_points(tmp_path: pathlib.Path) -> None:
    config_dir = tmp_path
    env_dir = config_dir / "env"
    strategy_dir = config_dir / "strategy"
    fees_dir = config_dir / "fees"

    env_dir.mkdir()
    strategy_dir.mkdir()
    fees_dir.mkdir()

    (env_dir / "custom.yaml").write_text(
        """
name: custom
fees_profile: custom_fee
""".strip()
    )

    (strategy_dir / "custom.yaml").write_text("name: custom\n")

    (fees_dir / "custom_fee.yaml").write_text(
        """
name: custom_fee
vip_tier: 0
maker_bps: 8
taker_bps: 10
notes: Sample configuration expressed in basis points.
""".strip()
    )

    bundle = load_config("custom", "custom", config_dir=config_dir)

    assert math.isclose(bundle.fees["maker"], 0.0008, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(bundle.fees["taker"], 0.001, rel_tol=0, abs_tol=1e-9)
    assert bundle.fees["vip_tier"] == 0


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
