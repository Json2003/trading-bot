"""Unified data feed abstraction built on top of ccxt-style clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
import logging

from .atr import compute_atr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketInstrument:
    """Descriptor of a market stream to subscribe to."""

    venue: str
    symbol: str
    timeframe: str | None = None
    alias: str | None = None

    def key(self) -> str:
        """Return the canonical identifier for the instrument."""

        return self.alias or f"{self.venue}:{self.symbol}"


@dataclass(frozen=True)
class OHLCV:
    """Simple OHLCV datapoint."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketData:
    """Snapshot of market information for an instrument."""

    venue: str
    symbol: str
    timestamp: datetime
    price: float
    ohlcv: tuple[OHLCV, ...] = field(default_factory=tuple)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.symbol}"

    def latest_candle(self) -> OHLCV | None:
        """Return the most recent OHLCV candle when available."""

        return self.ohlcv[-1] if self.ohlcv else None

    def atr(self, period: int = 14) -> float | None:
        """Compute an Average True Range value from the cached candles."""

        if not self.ohlcv:
            return None
        return compute_atr(self.ohlcv, period)


class UnifiedDataFeed:
    """Collects market data across venues via ccxt compatible clients."""

    def __init__(
        self,
        clients: Mapping[str, Any],
        instruments: Iterable[MarketInstrument],
        *,
        ohlcv_candles: int = 0,
        default_timeframe: str = "1m",
        log: logging.Logger | None = None,
    ) -> None:
        if not clients:
            raise ValueError("At least one client must be supplied")
        self._clients = dict(clients)
        self._instruments = tuple(instruments)
        if not self._instruments:
            raise ValueError("At least one instrument must be supplied")
        self._ohlcv_candles = int(max(0, ohlcv_candles))
        self._default_timeframe = default_timeframe
        self._log = log or logger

    def fetch(self) -> dict[str, MarketData]:
        """Fetch the latest snapshot for all configured instruments."""

        snapshots: dict[str, MarketData] = {}
        for instrument in self._instruments:
            client = self._clients.get(instrument.venue)
            if client is None:
                raise KeyError(f"No client configured for venue {instrument.venue!r}")

            ticker = client.fetch_ticker(instrument.symbol)
            timestamp = self._extract_timestamp(ticker)
            price = self._extract_price(ticker)

            ohlcv: tuple[OHLCV, ...] = ()
            raw_ohlcv: list[Any] | None = None
            if self._ohlcv_candles and hasattr(client, "fetch_ohlcv"):
                tf = instrument.timeframe or self._default_timeframe
                try:
                    raw_ohlcv = client.fetch_ohlcv(
                        instrument.symbol, tf, limit=self._ohlcv_candles
                    )
                except Exception as exc:  # pragma: no cover - ccxt errors are runtime specific
                    self._log.warning(
                        "Failed to fetch OHLCV for %s on %s: %s",
                        instrument.symbol,
                        instrument.venue,
                        exc,
                    )
                else:
                    ohlcv = tuple(self._transform_ohlcv(raw_ohlcv))

            key = instrument.key()
            snapshots[key] = MarketData(
                venue=instrument.venue,
                symbol=instrument.symbol,
                timestamp=timestamp,
                price=price,
                ohlcv=ohlcv,
                raw={"ticker": ticker, "ohlcv": raw_ohlcv} if raw_ohlcv is not None else {"ticker": ticker},
            )
        return snapshots

    @staticmethod
    def _extract_timestamp(ticker: Mapping[str, Any]) -> datetime:
        value = ticker.get("timestamp") or ticker.get("datetime")
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000 if value > 10_000_000_000 else value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _extract_price(ticker: Mapping[str, Any]) -> float:
        for key in ("last", "close", "bid", "ask"):
            value = ticker.get(key)
            if value is not None:
                return float(value)
        raise ValueError("Ticker payload does not contain a usable price")

    @staticmethod
    def _transform_ohlcv(raw: Iterable[Iterable[Any]]) -> Iterable[OHLCV]:
        for candle in raw:
            if len(candle) < 6:
                continue
            ts, o, h, l, c, v = candle[:6]
            if isinstance(ts, (int, float)):
                timestamp = datetime.fromtimestamp(ts / 1000 if ts > 10_000_000_000 else ts, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            yield OHLCV(
                timestamp=timestamp,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
            )


__all__ = ["MarketInstrument", "OHLCV", "MarketData", "UnifiedDataFeed"]
