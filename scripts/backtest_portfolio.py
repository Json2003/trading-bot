"""Compatibility wrapper for the multi-strategy portfolio backtest.

This script restores the legacy ``scripts/backtest_portfolio.py`` entry point
referenced throughout the documentation.  It simply delegates to
``multi_strategy_backtest.run_backtest`` so the existing behavior is preserved
without code duplication.
"""

from __future__ import annotations

from multi_strategy_backtest import run_backtest


def main() -> None:
    """Execute the delegated backtest."""

    run_backtest()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
