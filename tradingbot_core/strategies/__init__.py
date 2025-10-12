"""Strategy implementations built on the lightweight core protocol."""

from .arbitrage import CrossExArb
from .dca import DCAMartingale
from .grid import GridConfig, GridStrategy
from .momentum import MomentumEMA

__all__ = [
    "CrossExArb",
    "DCAMartingale",
    "GridConfig",
    "GridStrategy",
    "MomentumEMA",
]
