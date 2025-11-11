#!/usr/bin/env python3
"""Lightweight readiness checker used in documentation and quickstarts."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

try:  # Optional dependency: handled gracefully in ``check_readiness``
    import pandas as pd
except Exception:  # pragma: no cover - lazy import fallback
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _run_command(command: Iterable[str], verbose: bool) -> None:
    """Run *command* capturing output unless ``verbose`` is true."""

    subprocess.run(
        list(command),
        check=False,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )


def _repo_root() -> Path:
    """Return the repository root based on the script location."""

    return Path(__file__).resolve().parent.parent


def check_readiness(fix_issues: bool = False, verbose: bool = False) -> List[str]:
    """Run a small subset of the full readiness checks.

    Parameters
    ----------
    fix_issues:
        Attempt to remediate problems using helper scripts when ``True``.
    verbose:
        Emit logging output explaining which checks failed.
    """

    if verbose:
        logging.basicConfig(level=logging.INFO)

    repo_root = _repo_root()
    issues: List[str] = []

    # Python version --------------------------------------------------------
    if sys.version_info < (3, 8):
        issues.append(f"Python {sys.version.split()[0]} < 3.8")

    # Virtual environment ---------------------------------------------------
    venv_path = repo_root / "venv"
    if not venv_path.exists():
        issues.append("Virtual environment missing")
        if fix_issues:
            install_script = repo_root / "install.sh"
            if install_script.exists():
                logger.info("Creating venv with install.sh")
                _run_command([str(install_script)], verbose)

    # Dependency availability ----------------------------------------------
    try:
        import pandas  # type: ignore  # noqa: F401
        import ta  # type: ignore  # noqa: F401
        from dateutil import parser as _dateutil_parser  # noqa: F401
    except ImportError as exc:
        issues.append(f"Dependency missing: {exc}")
        if fix_issues:
            requirements = repo_root / "tradingbot_ibkr" / "requirements.txt"
            if requirements.exists():
                logger.info("Installing dependencies")
                _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)], verbose)
    else:
        # Version checks only if imports succeeded
        try:
            pandas_version = tuple(int(part) for part in pandas.__version__.split(".")[:3])  # type: ignore[name-defined]
            if pandas_version < (2, 0, 0):
                issues.append(f"pandas version {pandas.__version__} < 2.0.0")
        except Exception:  # pragma: no cover - defensive
            pass

        try:
            ta_version = tuple(int(part) for part in ta.__version__.split(".")[:3])  # type: ignore[name-defined]
            if ta_version < (0, 10, 2):
                issues.append(f"ta version {ta.__version__} < 0.10.2")
        except Exception:  # pragma: no cover - defensive
            pass

    # Critical file checks --------------------------------------------------
    critical_files = [
        repo_root / "backtest" / "sample_data" / "sample_ohlcv.csv",
        repo_root / "backtest" / "strategies" / "sma_filtered.py",
        repo_root / "scripts" / "run_backtest.py",
    ]

    for file_path in critical_files:
        if not file_path.exists():
            issues.append(f"File missing: {file_path.relative_to(repo_root)}")
            if fix_issues and file_path.name == "sample_ohlcv.csv":
                logger.info("Creating sample data")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                generator = repo_root / "scripts" / "generate_sample_data.py"
                if generator.exists():
                    _run_command([sys.executable, str(generator)], verbose)

    # Sample data quality ---------------------------------------------------
    sample_path = repo_root / "backtest" / "sample_data" / "sample_ohlcv.csv"
    if sample_path.exists():
        if pd is None:
            issues.append("pandas required to validate sample data")
        else:
            try:
                df = pd.read_csv(sample_path, parse_dates=["timestamp"])
            except Exception as exc:
                issues.append(f"Sample data invalid: {exc}")
            else:
                required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
                if not required_columns.issubset(df.columns):
                    issues.append("Sample data missing required columns")
                if len(df) < 144:
                    issues.append(f"Sample data has {len(df)} rows, need >=144 for SMA")
                    if fix_issues:
                        logger.info("Regenerating sample data")
                        generator = repo_root / "scripts" / "generate_sample_data.py"
                        if generator.exists():
                            _run_command([sys.executable, str(generator)], verbose)

    if verbose:
        if issues:
            for issue in issues:
                logger.info("Issue: %s", issue)
        else:
            logger.info("All checks passed.")

    return issues


def main() -> int:
    """Command line entry point for the lightweight checker."""

    parser = argparse.ArgumentParser(description="Run lightweight readiness checks")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to automatically remediate issues")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic information")
    args = parser.parse_args()

    issues = check_readiness(args.fix_issues, args.verbose)
    return 0 if not issues else 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
