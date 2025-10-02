"""Application configuration helpers and settings models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# The repository root is two levels up from this file (``src/settings.py``).
ROOT = Path(__file__).resolve().parents[1]


class AppSettings(BaseSettings):
    """Typed runtime settings sourced from environment variables."""

    TB_MODE: Literal["paper", "live"] = "paper"
    BROKER: str = "IBKR"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=ROOT / ".env", case_sensitive=False)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    path:
        Absolute path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed contents of the YAML file. Empty files return an empty dict.
    """

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Configuration file must contain a mapping: {path}")

    return data


def load_env_config(mode: str) -> dict[str, Any]:
    """Return environment configuration for the requested trading mode."""

    return load_yaml(ROOT / "config" / "env" / f"{mode}.yaml")


def load_strategy_config(name: str) -> dict[str, Any]:
    """Return the strategy configuration with the supplied identifier."""

    return load_yaml(ROOT / "config" / "strategy" / f"{name}.yaml")


__all__ = ["AppSettings", "load_env_config", "load_strategy_config"]

