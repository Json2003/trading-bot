#!/usr/bin/env python3
"""Lightweight trading bot readiness checker.

This utility provides a very small subset of the full
``check_trading_readiness.py`` script so users can run a quick diagnostic
without pulling in the larger dependency tree.  The checker focuses on a
few high-level items:

* The local virtual environment exists.
* Core dependencies (``pandas`` and ``ta``) can be imported.
* Sample OHLCV data is available for quick backtests.
* A ``.env`` file exists to configure credentials and risk flags.

When the ``--fix-issues`` flag is supplied the checker will attempt to
resolve the problems by calling existing project tooling, installing
dependencies, or generating placeholder files.  The function returns the
list of issues that still require manual attention so it can be reused
programmatically by other tools.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_command(command: List[str], verbose: bool) -> None:
    """Execute *command* while optionally forwarding stdout/stderr."""

    subprocess.run(command, check=False, stdout=None if verbose else subprocess.DEVNULL)


def _ensure_file(path: Path, contents: str, verbose: bool) -> None:
    """Create *path* with *contents* if it does not yet exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    if verbose:
        print(f"Created {path}")


def check_readiness(fix_issues: bool = False, verbose: bool = False) -> List[str]:
    """Run a lightweight readiness check for the repository.

    Parameters
    ----------
    fix_issues:
        When ``True`` the checker attempts to remediate problems using the
        project's helper scripts and by creating placeholder files.
    verbose:
        Emit status messages explaining the checks being performed.

    Returns
    -------
    list[str]
        A list of outstanding issues that require manual resolution.
    """

    repo_root = Path(__file__).resolve().parent.parent
    issues: List[str] = []

    # 1. Virtual environment -------------------------------------------------
    venv_path = repo_root / "venv"
    if not venv_path.exists():
        issues.append("venv missing")
        if fix_issues:
            install_script = repo_root / "install.sh"
            if install_script.exists():
                if verbose:
                    print("Attempting to create virtual environment via install.sh")
                _run_command([str(install_script)], verbose)
            else:
                if verbose:
                    print("install.sh not found; unable to auto-create virtual environment")

    # 2. Dependencies --------------------------------------------------------
    try:
        import pandas  # type: ignore  # noqa: F401
        import ta  # type: ignore  # noqa: F401
    except ImportError:
        issues.append("Dependencies missing")
        if fix_issues:
            requirements = repo_root / "tradingbot_ibkr" / "requirements.txt"
            if requirements.exists():
                if verbose:
                    print("Installing Python dependencies from requirements.txt")
                _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)], verbose)
            elif verbose:
                print("requirements.txt not found; skipping dependency installation")

    # 3. Sample data ---------------------------------------------------------
    sample_path = repo_root / "backtest" / "sample_data" / "sample_ohlcv.csv"
    if not sample_path.exists():
        issues.append("Sample data missing")
        if fix_issues:
            if verbose:
                print("Creating placeholder sample OHLCV data")
            _ensure_file(sample_path, "# Paste the sample CSV content here\n", verbose)

    # 4. Configuration file --------------------------------------------------
    env_file = repo_root / ".env"
    if not env_file.exists():
        issues.append(".env missing")
        if fix_issues:
            if verbose:
                print("Creating example .env file")
            _ensure_file(env_file, "API_KEY=example\n", verbose)

    # Stubbed checks to keep the interface consistent with the more
    # comprehensive readiness checker.  They are intentionally lightweight so
    # this script stays dependency-free.
    if verbose:
        print("Checking data quality... OK (stub)")
        print("Checking models... OK (stub)")
        print("Checking risk settings... OK (stub)")

    if issues:
        print("Issues found:", issues)
    else:
        print("All checks passed.")

    return issues


def main() -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Run a lightweight readiness check")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to auto-fix common issues")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    args = parser.parse_args()

    issues = check_readiness(args.fix_issues, args.verbose)
    return 0 if not issues else 1


if __name__ == "__main__":  # pragma: no cover - convenience script
    raise SystemExit(main())
