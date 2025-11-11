"""Strategy implementations built on the lightweight core protocol.

Expose commonly used strategy classes at package level for convenient imports
like ``from tradingbot_core.strategies import DCAMartingale``.
"""

from .cross_exchange_arb import CrossExchangeArbitrage
from .grid import GridConfig, GridStrategy
from .momentum import MomentumEMA
from .dca import DCAMartingale

__all__ = [
	"CrossExchangeArbitrage",
	"GridConfig",
	"GridStrategy",
	"MomentumEMA",
	"DCAMartingale",
]
