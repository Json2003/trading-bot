"""Unified data feed abstraction built on top of ccxt-style clients."""

from __future__ import annotations

from collections import deque

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
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
    session: str | None = None
    ohlcv: tuple[OHLCV, ...] = field(default_factory=tuple)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    _atr_cache: dict[int, float] = field(default_factory=dict, repr=False)

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
        cached = self._atr_cache.get(period)
        if cached is not None:
            return cached
        if len(self.ohlcv) < max(2, period):
            return None
        value = compute_atr(self.ohlcv, period)
        self._atr_cache[period] = value
        return value


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
        self._ohlcv_cache: dict[str, deque[OHLCV]] = {}
        self._open_interest_cache: dict[str, float] = {}

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
            key = instrument.key()
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
                    transformed = tuple(self._transform_ohlcv(raw_ohlcv))
                    if transformed:
                        cache = self._ohlcv_cache.setdefault(
                            key, deque(maxlen=self._ohlcv_candles or None)
                        )
                        for candle in transformed:
                            if cache and cache[-1].timestamp >= candle.timestamp:
                                # Replace out-of-order or duplicate entries with the latest data
                                if cache[-1].timestamp == candle.timestamp:
                                    cache[-1] = candle
                                continue
                            cache.append(candle)
                        ohlcv = tuple(cache)
                    else:
                        ohlcv = tuple(self._ohlcv_cache.get(key, ()))
            else:
                ohlcv = tuple(self._ohlcv_cache.get(key, ()))

            session_label = self._infer_session(timestamp)
            funding_payload: Any | None = None
            open_interest_payload: Any | None = None
            funding_rate = None
            open_interest = None
            open_interest_change = None

            fetch_funding = getattr(client, "fetch_funding_rate", None)
            if callable(fetch_funding):
                try:
                    funding_payload = fetch_funding(instrument.symbol)
                except Exception as exc:  # pragma: no cover - runtime dependent
                    self._log.debug(
                        "Failed to fetch funding for %s on %s: %s",
                        instrument.symbol,
                        instrument.venue,
                        exc,
                    )
                else:
                    funding_rate = self._extract_funding_rate(funding_payload)

            fetch_oi = getattr(client, "fetch_open_interest", None)
            if callable(fetch_oi):
                try:
                    open_interest_payload = fetch_oi(instrument.symbol)
                except Exception as exc:  # pragma: no cover - runtime dependent
                    self._log.debug(
                        "Failed to fetch open interest for %s on %s: %s",
                        instrument.symbol,
                        instrument.venue,
                        exc,
                    )
                else:
                    open_interest = self._extract_open_interest(open_interest_payload)
                    if open_interest is not None:
                        prev = self._open_interest_cache.get(key)
                        if prev is not None:
                            open_interest_change = open_interest - prev
                        self._open_interest_cache[key] = open_interest

            metrics: dict[str, Any] = {}
            if session_label:
                metrics["session"] = session_label
            if funding_rate is not None:
                metrics["funding_rate"] = funding_rate
            if open_interest is not None:
                metrics["open_interest"] = open_interest
            if open_interest_change is not None:
                metrics["open_interest_change"] = open_interest_change

            raw_payload: dict[str, Any] = {"ticker": ticker}
            if raw_ohlcv is not None:
                raw_payload["ohlcv"] = raw_ohlcv
            if funding_payload is not None:
                raw_payload["funding_rate"] = funding_payload
            if open_interest_payload is not None:
                raw_payload["open_interest"] = open_interest_payload

            snapshots[key] = MarketData(
                venue=instrument.venue,
                symbol=instrument.symbol,
                timestamp=timestamp,
                price=price,
                session=session_label,
                ohlcv=ohlcv,
                metrics=metrics,
                raw=raw_payload,
            )
        return snapshots


    @staticmethod
    def _infer_session(timestamp: datetime) -> str:
        ts = timestamp.astimezone(timezone.utc)
        hour = ts.hour
        if 0 <= hour < 8:
            return "asia"
        if 8 <= hour < 16:
            return "europe"
        return "us"

    @staticmethod
    def _extract_funding_rate(payload: Any) -> float | None:
        if isinstance(payload, Mapping):
            for key in ("fundingRate", "funding_rate", "rate", "value"):
                value = payload.get(key)
                if value is None and isinstance(payload.get("info"), Mapping):
                    value = payload["info"].get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_open_interest(payload: Any) -> float | None:
        if isinstance(payload, Mapping):
            for key in (
                "openInterest",
                "open_interest",
                "openInterestAmount",
                "openInterestValue",
                "value",
            ):
                value = payload.get(key)
                if value is None and isinstance(payload.get("info"), Mapping):
                    value = payload["info"].get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

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


class ReplayDataFeed:
    """Replay a pre-recorded sequence of :class:`MarketData` snapshots."""

    def __init__(
        self,
        snapshots: Mapping[str, Sequence[MarketData]],
    ) -> None:
        if not snapshots:
            raise ValueError("snapshots must contain at least one instrument")

        lengths = {len(series) for series in snapshots.values()}
        if not lengths or 0 in lengths:
            raise ValueError("each instrument must provide at least one datapoint")
        if len(lengths) != 1:
            raise ValueError("all instruments must have the same number of snapshots")

        self._snapshots: dict[str, Sequence[MarketData]] = {
            key: tuple(series) for key, series in snapshots.items()
        }
        self._length = lengths.pop()
        self._cursor = 0

    def reset(self) -> None:
        """Seek back to the first snapshot."""

        self._cursor = 0

    @property
    def remaining(self) -> int:
        """Return the number of snapshots left to replay."""

        return self._length - self._cursor

    def fetch(self) -> dict[str, MarketData]:
        """Return the next snapshot in the replay sequence."""

        if self._cursor >= self._length:
            raise StopIteration("no more market data to replay")

        frame = {
            key: series[self._cursor]
            for key, series in self._snapshots.items()
        }
        self._cursor += 1
        return frame


__all__ = ["MarketInstrument", "OHLCV", "MarketData", "UnifiedDataFeed", "ReplayDataFeed"]
