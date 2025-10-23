"""Lightweight adapter around ccxt exchanges to fetch recent OHLCV bars."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Mapping, Tuple
import logging

try:  # pragma: no cover - optional import for typing only
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - the package might not be installed during tests
    ccxt = None  # type: ignore

from backtest.indicators import ATR
from tradingbot_core.strategy import Bar

logger = logging.getLogger(__name__)


BarTuple = Tuple[int, float, float, float, float, float]


@dataclass(frozen=True)
class OHLCVBar:
    """Internal representation of an OHLCV candle returned by ccxt."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_sequence(cls, data: Iterable[float]) -> "OHLCVBar":
        values = list(data)[:6]
        if len(values) < 6:
            raise ValueError("OHLCV payload must contain at least six values")
        ts, o, h, l, c, v = values
        return cls(
            ts=int(ts),
            open=float(o),
            high=float(h),
            low=float(l),
            close=float(c),
            volume=float(v),
        )


class CCXTFeed:
    """Fetches the latest bars for a set of symbols across ccxt exchanges."""

    def __init__(
        self,
        exchanges: Mapping[str, "ccxt.Exchange" | object],
        symbols: Iterable[str],
        timeframe: str = "1m",
        *,
        atr_window: int = 14,
        log: logging.Logger | None = None,
    ) -> None:
        if not exchanges:
            raise ValueError("At least one exchange must be provided")
        if not symbols:
            raise ValueError("At least one symbol must be provided")
        if atr_window <= 0:
            raise ValueError("atr_window must be positive")

        self._exchanges: Dict[str, object] = dict(exchanges)
        self._symbols = tuple(symbols)
        self._timeframe = timeframe
        self._log = log or logger
        self._atr_window = atr_window
        self._history: Dict[str, Deque[BarTuple]] = {
            symbol: deque(maxlen=self._atr_window + 3) for symbol in self._symbols
        }
        self._atr: Dict[str, ATR] = {symbol: ATR(window=self._atr_window) for symbol in self._symbols}

    def latest_bars(self) -> Dict[str, Bar]:
        """Fetch the most recent OHLCV bar for each symbol on every exchange."""

        bars: Dict[str, Bar] = {}
        for ex_name, exchange in self._exchanges.items():
            for symbol in self._symbols:
                try:
                    raw = getattr(exchange, "fetch_ohlcv")(symbol, timeframe=self._timeframe, limit=2)
                except Exception as exc:  # pragma: no cover - ccxt errors depend on runtime conditions
                    self._log.warning("Failed to fetch OHLCV for %s on %s: %s", symbol, ex_name, exc)
                    continue

                if not raw:
                    continue

                try:
                    bar = OHLCVBar.from_sequence(raw[-1])
                except ValueError as exc:
                    self._log.warning(
                        "Failed to parse OHLCV for %s on %s: %s", symbol, ex_name, exc
                    )
                    continue

                bar_tuple: BarTuple = (
                    bar.ts,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                )
                history = self._history.setdefault(
                    symbol, deque(maxlen=self._atr_window + 3)
                )
                history.append(bar_tuple)
                indicator = self._atr.setdefault(symbol, ATR(window=self._atr_window))
                indicator.update(bar_tuple)
                key = f"{ex_name}:{symbol}"
                converted_bar = Bar(
                    bar.ts, bar.open, bar.high, bar.low, bar.close, bar.volume
                )
                bars[key] = converted_bar

                existing = bars.get(symbol)
                if existing is None or existing.ts <= bar.ts:
                    bars[symbol] = converted_bar

        return bars

    def atr(self, symbol: str) -> float | None:
        """Return the most recent Average True Range for ``symbol``."""

        indicator = self._atr.get(symbol)
        if indicator is None:
            return None
        return indicator.value

    def history(self, symbol: str, limit: int | None = None) -> list[BarTuple]:
        """Return cached OHLCV history for ``symbol``.

        Parameters
        ----------
        symbol:
            Market symbol whose cached bars should be returned.
        limit:
            Optional cap on how many of the most recent bars to return. Non-positive
            values yield an empty list.
        """

        hist = self._history.get(symbol)
        if not hist:
            return []
        if limit is None:
            return list(hist)
        if limit <= 0:
            return []
        return list(hist)[-limit:]


__all__ = ["CCXTFeed"]
