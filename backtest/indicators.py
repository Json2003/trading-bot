"""Lightweight technical indicators for streaming backtests."""

from __future__ import annotations

from collections import deque
from typing import Deque, Sequence, Tuple


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

