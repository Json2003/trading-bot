"""Lightweight technical indicators for streaming backtests."""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, Dict, Sequence, Tuple


def _extract_hlc(bar: object) -> Tuple[float, float, float]:
    """Return the high/low/close values from a tuple-like bar or object."""

    if isinstance(bar, Sequence) and not isinstance(bar, (str, bytes, bytearray)):
        if len(bar) < 5:
            raise ValueError("expected (ts, open, high, low, close, volume) sequence")
        return float(bar[2]), float(bar[3]), float(bar[4])

    try:
        high = getattr(bar, "high")
        low = getattr(bar, "low")
        close = getattr(bar, "close")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError("bar must be tuple-like or expose high/low/close attributes") from exc

    return float(high), float(low), float(close)


def true_range(prev_close: float, high: float, low: float) -> float:
    """Wilder's True Range for the current bar."""

    return max(high - low, abs(high - prev_close), abs(low - prev_close))


class ATR:
    """Rolling Average True Range (Wilder) with constant-time updates."""

    def __init__(self, window: int = 14) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self._trs: Deque[float] = deque(maxlen=self.window)
        self.prev_close: float | None = None
        self.value: float | None = None

    def update(self, bar: object) -> float | None:
        """Ingest a bar and return the updated ATR value."""

        high, low, close = _extract_hlc(bar)
        if self.prev_close is None:
            tr = float(high - low)
        else:
            tr = true_range(self.prev_close, high, low)
        self.prev_close = float(close)

        if len(self._trs) < self.window:
            self._trs.append(tr)
            self.value = sum(self._trs) / len(self._trs)
        else:
            prev = self.value if self.value is not None else tr
            self.value = (prev * (self.window - 1) + tr) / self.window
        return self.value


__all__ = ["ATR", "true_range"]


class RollingBeta:
    """Rolling OLS beta of each asset versus a reference market series."""

    def __init__(self, window: int = 240) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        self.window = int(window)
        self._pairs: Dict[str, Deque[tuple[float, float]]] = {}
        self.latest: Dict[str, float] = {}

    def update(self, symbol: str, r_asset: float, r_market: float) -> float | None:
        """Update the rolling beta for ``symbol`` and return the new estimate."""

        dq = self._pairs.setdefault(symbol, deque(maxlen=self.window))
        dq.append((float(r_asset), float(r_market)))

        min_obs = max(30, self.window // 5)
        if len(dq) < min_obs:
            return None

        returns = np.asarray(dq, dtype=float)
        asset = returns[:, 0]
        market = returns[:, 1]

        cov = np.cov(asset, market, ddof=1)
        var_market = float(cov[1, 1])
        beta = float(cov[0, 1] / var_market) if var_market > 1e-12 else 0.0

        self.latest[symbol] = beta
        return beta


__all__.append("RollingBeta")

