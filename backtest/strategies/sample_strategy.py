"""Sample strategy module providing generate_signals for dynamic loading.

This proxies to backtest.signals.generate_signals.
"""
from ..signals import generate_signals  # re-export

__all__ = ["generate_signals"]
