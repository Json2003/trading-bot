"""Dependency-free safety preflight for the local paper deployment.

This checker validates configuration only. It never contacts an exchange,
places orders, changes risk limits, or creates credentials.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
TRUTHY = {"1", "true", "yes", "on"}
PLACEHOLDER_MARKERS = ("replace", "your-", "changeme", "example", "token_here")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def validate(config_path: Path, env_file: Path | None, require_token: bool) -> list[str]:
    errors: list[str] = []
    values = dict(read_env_file(env_file)) if env_file else {}
    for key in ("TRADING_OPERATOR_MODE", "TRADING_OPERATOR_RUNTIME", "TRADING_OPERATOR_HOST", "TRADING_OPERATOR_TOKEN"):
        values.setdefault(key, os.getenv(key, ""))

    if (values.get("TRADING_OPERATOR_MODE") or "paper").lower() != "paper":
        errors.append("TRADING_OPERATOR_MODE must be paper")
    if (values.get("TRADING_OPERATOR_RUNTIME") or "synthetic-smoke").lower() not in {"synthetic-smoke", "synthetic", "none", "disabled"}:
        errors.append("TRADING_OPERATOR_RUNTIME must be synthetic-smoke, synthetic, none, or disabled")
    if (values.get("TRADING_OPERATOR_HOST") or "127.0.0.1") not in LOOPBACK_HOSTS:
        errors.append("TRADING_OPERATOR_HOST must remain loopback-only")

    token = values.get("TRADING_OPERATOR_TOKEN", "")
    if require_token and (len(token) < 32 or any(marker in token.lower() for marker in PLACEHOLDER_MARKERS)):
        errors.append("TRADING_OPERATOR_TOKEN must be a locally generated token of at least 32 characters")

    text = config_path.read_text(encoding="utf-8").lower()
    required = {
        "profile: paper": "config profile must be paper",
        "live_trading: false": "live_trading must be false",
        "allow_live_orders: false": "allow_live_orders must be false",
        "host: 127.0.0.1": "operator host must be loopback",
        "kill_switch_drawdown_fraction: 0.02": "paper drawdown kill switch must be 2%",
        "auto_rearm: true": "paper recovery auto-rearm must be enabled",
        "require_flat: true": "paper recovery must require a flat broker",
        "full_reset_requires_human_approval: true": "full recovery reset must require human approval",
    }
    for marker, message in required.items():
        if marker not in text:
            errors.append(message)
    if "testnet: false" in text or "api_key:" in text or "secret:" in text:
        errors.append("paper deployment config must not contain live exchange settings or credentials")

    for key, value in values.items():
        if "LIVE" in key.upper() and value.lower() in TRUTHY:
            errors.append(f"live-enabling variable is set: {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper-deployment.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--require-token", action="store_true")
    args = parser.parse_args()
    try:
        errors = validate(args.config, args.env_file, args.require_token)
    except (OSError, ValueError) as exc:
        print(f"PAPER PREFLIGHT: FAIL: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("PAPER PREFLIGHT: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PAPER PREFLIGHT: PASS — loopback, paper-only, credential-free profile validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
