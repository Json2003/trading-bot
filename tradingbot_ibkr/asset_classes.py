from __future__ import annotations
"""Enumeration of supported asset classes for the trading bot."""
from enum import Enum

class AssetClass(Enum):
    """Supported asset classes for trading operations."""
    FOREX = "forex"
    OPTIONS = "options"
    FUTURES = "futures"
    CRYPTO = "crypto"
    STOCKS = "stocks"

# Default volatility thresholds used by the trading engine for each asset class.
VOLATILITY_THRESHOLDS = {
    AssetClass.FOREX: 0.02,
    AssetClass.OPTIONS: 0.10,
    AssetClass.FUTURES: 0.04,
    AssetClass.CRYPTO: 0.05,
    AssetClass.STOCKS: 0.03,
}


def get_volatility_threshold(asset_class: AssetClass) -> float:
    """Return the default volatility threshold for the given asset class."""
    return VOLATILITY_THRESHOLDS.get(asset_class, 0.05)
