"""Tests for convenience configuration loader helpers."""

from tradingbot_core.config import load_env_config, load_strategy_config


def test_env_load_includes_identifier() -> None:
    cfg = load_env_config("paper")

    assert cfg["env"] == "paper"
    assert cfg["name"] == "paper"


def test_strategy_load_contains_expected_fields() -> None:
    strategy = load_strategy_config("sample_meanrev")

    assert strategy["name"] == "sample_meanrev"
    assert "entry_z" in strategy
