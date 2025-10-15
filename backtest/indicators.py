"""Lightweight technical indicators used by the backtest engine."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Protocol, Tuple


class _BarLike(Protocol):
    """Protocol describing the minimal bar attributes used by :class:`ATR`."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


BarInput = Tuple[int, float, float, float, float, float]


def true_range(prev_close: float, high: float, low: float) -> float:
    """Return the True Range value for the provided prices."""

    return max(high - low, abs(high - prev_close), abs(low - prev_close))


class ATR:
    """Rolling Average True Range (Wilder) with ``O(1)`` updates."""

    def __init__(self, window: int = 14):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self._trs: Deque[float] = deque(maxlen=window)
        self.prev_close: float | None = None
        self.value: float | None = None

    def _get_bar_values(self, bar: _BarLike | BarInput) -> Tuple[float, float, float]:
        if isinstance(bar, tuple):
            _ts, _o, high, low, close, _volume = bar
        else:
            high = float(bar.high)
            low = float(bar.low)
            close = float(bar.close)
        return high, low, close

    def update(self, bar: _BarLike | BarInput) -> float | None:
        """Update the ATR with a new bar and return the current value."""

        high, low, close = self._get_bar_values(bar)
        if self.prev_close is None:
            tr = high - low
        else:
            tr = true_range(self.prev_close, high, low)
        self.prev_close = close

        if len(self._trs) < self.window:
            self._trs.append(tr)
            self.value = sum(self._trs) / len(self._trs)
        else:
            # Wilder smoothing
            assert self.value is not None
            self.value = (self.value * (self.window - 1) + tr) / self.window
        return self.value

    def warmup(self, bars: Iterable[_BarLike | BarInput]) -> float | None:
        """Convenience helper to seed the ATR with historical data."""

        value: float | None = None
        for bar in bars:
            value = self.update(bar)
        return value


__all__ = ["ATR", "true_range"]

