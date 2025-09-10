"""Shim module mapping adaptive.regimes to backtest.regime functions."""
from ..regime import classify_regime, realized_vol, trend_slope

__all__ = ["classify_regime", "realized_vol", "trend_slope"]
