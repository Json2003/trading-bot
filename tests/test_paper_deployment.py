from __future__ import annotations

from pathlib import Path

from scripts.validate_paper_deployment import validate


def test_paper_profile_passes_without_runtime_environment(monkeypatch) -> None:
    for key in (
        "TRADING_OPERATOR_MODE",
        "TRADING_OPERATOR_RUNTIME",
        "TRADING_OPERATOR_HOST",
        "TRADING_OPERATOR_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    errors = validate(Path("configs/paper-deployment.yaml"), None, require_token=False)

    assert errors == []


def test_paper_profile_rejects_public_binding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRADING_OPERATOR_HOST", "0.0.0.0")
    errors = validate(Path("configs/paper-deployment.yaml"), None, require_token=False)
    assert "TRADING_OPERATOR_HOST must remain loopback-only" in errors


def test_paper_profile_rejects_placeholder_token(tmp_path: Path) -> None:
    env_file = tmp_path / "paper.env"
    env_file.write_text(
        "TRADING_OPERATOR_MODE=paper\n"
        "TRADING_OPERATOR_RUNTIME=synthetic-smoke\n"
        "TRADING_OPERATOR_HOST=127.0.0.1\n"
        "TRADING_OPERATOR_TOKEN=REPLACE_LOCALLY_WITH_A_RANDOM_TOKEN\n",
        encoding="utf-8",
    )
    errors = validate(Path("configs/paper-deployment.yaml"), env_file, require_token=True)
    assert "TRADING_OPERATOR_TOKEN must be a locally generated token of at least 32 characters" in errors
