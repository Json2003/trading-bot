"""Convenience imports for built-in strategies."""

from .sample_strategy import generate_signals
from .sma_filtered import generate_signals as sma_filtered_signals
from .arbitrage import generate_basis_signals, ArbitrageConfig

__all__ = [
    "generate_signals",
    "sma_filtered_signals",
    "generate_basis_signals",
    "ArbitrageConfig",
]
