"""Convenience imports for built-in strategies."""

from .sample_strategy import generate_signals
from .sma_filtered import generate_signals as sma_filtered_signals
from .arbitrage import generate_basis_signals, ArbitrageConfig
from .sma_trend_rsi import generate_signals as sma_trend_rsi_signals
from .sma_rsi_filtered import generate_signals as sma_rsi_filtered_signals

__all__ = [
    "generate_signals",
    "sma_filtered_signals",
    "generate_basis_signals",
    "ArbitrageConfig",
    "sma_trend_rsi_signals",
    "sma_rsi_filtered_signals",
]
