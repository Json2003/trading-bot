#!/usr/bin/env python3
"""Launch the local operator API and Electron console as one managed session."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ELECTRON_DIR = ROOT / "dashboard" / "electron-app"
API_SCRIPT = ROOT / "scripts" / "run_operator_api.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runtime", default="synthetic-smoke", choices=["synthetic-smoke", "none"])
    parser.add_argument("--cycle-seconds", type=float, default=1.0)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def npm_command() -> str:
    executable = "npm.cmd" if os.name == "nt" else "npm"
    resolved = shutil.which(executable)
    if not resolved:
        raise RuntimeError("npm was not found; install Node.js before launching the console")
    return resolved


def wait_for_health(url: str, process: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"operator API exited during startup with code {process.returncode}")
        try:
            with urlopen(url, timeout=1.0) as response:  # noqa: S310 - fixed loopback URL
                if response.status == 200:
                    return
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"operator API did not become healthy: {last_error}")


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.cycle_seconds <= 0:
        raise ValueError("cycle-seconds must be positive")

    npm = npm_command()
    if not args.skip_install and not (ELECTRON_DIR / "node_modules" / "electron").exists():
        subprocess.run([npm, "ci"], cwd=ELECTRON_DIR, check=True)

    token = os.getenv("TRADING_OPERATOR_TOKEN") or secrets.token_urlsafe(48)
    base_url = f"http://127.0.0.1:{args.port}"
    child_env = os.environ.copy()
    child_env.update(
        {
            "TRADING_OPERATOR_TOKEN": token,
            "TRADING_OPERATOR_URL": base_url,
            "TRADING_OPERATOR_HOST": "127.0.0.1",
            "TRADING_OPERATOR_PORT": str(args.port),
            "TRADING_OPERATOR_RUNTIME": args.runtime,
            "TRADING_OPERATOR_CYCLE_SECONDS": str(args.cycle_seconds),
        }
    )

    api_process: subprocess.Popen[bytes] | None = None
    try:
        api_process = subprocess.Popen([sys.executable, str(API_SCRIPT)], cwd=ROOT, env=child_env)
        wait_for_health(f"{base_url}/health", api_process)
        desktop = subprocess.run([npm, "start"], cwd=ELECTRON_DIR, env=child_env, check=False)
        return int(desktop.returncode)
    finally:
        terminate(api_process)


if __name__ == "__main__":
    raise SystemExit(main())
