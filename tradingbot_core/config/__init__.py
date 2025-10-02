"""Configuration loader that stitches together environment, strategy, and fee profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import os

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _coerce_int(value: str | None) -> int | str | None:
    """Attempt to coerce a string to an integer when possible."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


@dataclass(frozen=True)
class ConfigBundle:
    """Bundle of related configuration sections used by trading services."""

    env: Mapping[str, Any]
    strategy: Mapping[str, Any]
    fees: Mapping[str, Any]
    runtime: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return the bundle as a serialisable dictionary."""

        return {
            "env": dict(self.env),
            "strategy": dict(self.strategy),
            "fees": dict(self.fees),
            "runtime": dict(self.runtime),
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration file must contain a mapping: {path}")
    return data


def _runtime_overrides() -> dict[str, Any]:
    runtime: dict[str, Any] = {}

    mode = os.getenv("TB_MODE")
    if mode:
        runtime["mode"] = mode

    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        runtime.setdefault("logging", {})["level"] = log_level

    broker_name = os.getenv("BROKER")
    if broker_name:
        broker: dict[str, Any] = {"name": broker_name}
        ibkr_base = os.getenv("IBKR_BASE_URL")
        ibkr_client = _coerce_int(os.getenv("IBKR_CLIENT_ID"))
        ibkr_account = os.getenv("IBKR_ACCOUNT_ID")
        if any(value is not None for value in (ibkr_base, ibkr_client, ibkr_account)):
            broker["ibkr"] = {
                key: value
                for key, value in {
                    "base_url": ibkr_base,
                    "client_id": ibkr_client,
                    "account_id": ibkr_account,
                }.items()
                if value is not None
            }
        runtime["broker"] = broker

    exchange_id = os.getenv("EXCHANGE_ID")
    if exchange_id:
        exchange: dict[str, Any] = {"id": exchange_id}
        api_key = os.getenv("EXCHANGE_API_KEY")
        secret = os.getenv("EXCHANGE_SECRET")
        if api_key:
            exchange["api_key"] = api_key
        if secret:
            exchange["secret"] = secret
        runtime["exchange"] = exchange

    return runtime


def load_config(env_name: str, strategy_name: str, *, config_dir: str | Path | None = None) -> ConfigBundle:
    """Load configuration sections for a given environment and strategy."""

    base_dir = Path(config_dir) if config_dir else _CONFIG_ROOT

    env_config = _load_yaml(base_dir / "env" / f"{env_name}.yaml")
    strategy_config = _load_yaml(base_dir / "strategy" / f"{strategy_name}.yaml")

    fees_profile = env_config.get("fees_profile") or strategy_config.get("fees_profile")
    fees_config: dict[str, Any] = {}
    if fees_profile:
        fees_config = _load_yaml(base_dir / "fees" / f"{fees_profile}.yaml")

    runtime = _runtime_overrides()

    return ConfigBundle(env=env_config, strategy=strategy_config, fees=fees_config, runtime=runtime)


__all__ = ["ConfigBundle", "load_config"]
