"""Lightweight adapter around ccxt exchanges to fetch recent OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping
import logging

from tradingbot_core.strategy import Bar

try:  # pragma: no cover - optional import for typing only
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - the package might not be installed during tests
    ccxt = None  # type: ignore

logger = logging.getLogger(__name__)


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
        log: logging.Logger | None = None,
    ) -> None:
        if not exchanges:
            raise ValueError("At least one exchange must be provided")
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        self._exchanges: Dict[str, object] = dict(exchanges)
        self._symbols = tuple(symbols)
        self._timeframe = timeframe
        self._log = log or logger

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
                key = f"{ex_name}:{symbol}"
                converted_bar = Bar(
                    bar.ts, bar.open, bar.high, bar.low, bar.close, bar.volume
                )
                bars[key] = converted_bar

                existing = bars.get(symbol)
                if existing is None or existing.ts <= bar.ts:
                    bars[symbol] = converted_bar

        return bars


__all__ = ["CCXTFeed"]
