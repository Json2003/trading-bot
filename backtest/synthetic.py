"""Helpers for generating synthetic market data for portfolio backtests."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Iterable
import math

import numpy as np

import pandas as pd

from engine.datafeed import MarketData, MarketInstrument, OHLCV, ReplayDataFeed, UnifiedDataFeed


def _base_price(symbol: str) -> float:
    symbol = symbol.upper()
    if "BTC" in symbol:
        return 60_000.0
    if "ETH" in symbol:
        return 2_000.0
    if "SOL" in symbol:
        return 120.0
    return 1_000.0


def _session_label(timestamp: datetime) -> str:
    return UnifiedDataFeed._infer_session(timestamp)  # type: ignore[attr-defined]


def build_synthetic_feed(
    instruments: Iterable[MarketInstrument],
    *,
    steps: int,
    seed: int = 7,
    timeframe: str = "1h",
    history: int = 120,
) -> tuple[ReplayDataFeed, pd.DataFrame]:
    """Return a :class:`ReplayDataFeed` populated with synthetic data."""

    instruments = list(instruments)
    if not instruments:
        raise ValueError("at least one instrument is required")
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(seed)
    horizon = timedelta(hours=1)
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1] or 1)
        horizon = timedelta(minutes=minutes)
    elif timeframe.endswith("h"):
        hours = int(timeframe[:-1] or 1)
        horizon = timedelta(hours=hours)

    start = datetime(2022, 1, 1, tzinfo=timezone.utc)

    # Generate base price paths per symbol so multi-venue instruments stay correlated.
    symbol_paths: dict[str, np.ndarray] = {}
    symbol_basis: dict[str, float] = {}
    for instrument in instruments:
        symbol = instrument.symbol
        if symbol in symbol_paths:
            continue
        base = _base_price(symbol)
        drift = 0.0004 if "ETH" in symbol else 0.0002
        vol = 0.01 if "BTC" in symbol else 0.008
        noise = rng.normal(drift, vol, size=steps)
        seasonal = [0.0025 * math.sin((idx / max(steps, 1)) * math.pi * 4) for idx in range(steps)]
        returns = [float(noise[idx]) + seasonal[idx] for idx in range(steps)]
        path: list[float] = []
        price = base
        for ret in returns:
            price *= math.exp(ret)
            path.append(price)
        symbol_paths[symbol] = path
        symbol_basis[symbol] = rng.normal(1.0, 0.002)

    alias_history: dict[str, deque[OHLCV]] = {
        instrument.key(): deque(maxlen=history)
        for instrument in instruments
    }

    snapshots: dict[str, list[MarketData]] = defaultdict(list)
    timeline: list[datetime] = []
    for step in range(steps):
        timestamp = start + horizon * step
        if not timeline:
            timeline = [start + horizon * idx for idx in range(steps)]
        session = _session_label(timestamp)
        for instrument in instruments:
            key = instrument.key()
            base_path = symbol_paths[instrument.symbol]
            base_price = base_path[step]
            alias_bias = rng.normal(0.0, 0.0015)
            if instrument.venue.lower() == "coinbase":
                alias_bias += 0.002 * math.sin(step / 10)
            price = base_price * (symbol_basis[instrument.symbol] + alias_bias)
            price = max(price, 1.0)

            prev = alias_history[key][-1].close if alias_history[key] else price
            open_px = prev
            shock = abs(rng.normal(0, 0.002))
            high = max(open_px, price) * (1 + shock)
            low = min(open_px, price) * (1 - shock)
            volume = abs(rng.normal(_base_price(instrument.symbol) * 0.01, 50.0))
            candle = OHLCV(timestamp=timestamp, open=open_px, high=high, low=low, close=price, volume=volume)
            alias_history[key].append(candle)

            snapshots[key].append(
                MarketData(
                    venue=instrument.venue,
                    symbol=instrument.symbol,
                    timestamp=timestamp,
                    price=price,
                    session=session,
                    ohlcv=tuple(alias_history[key]),
                    metrics={"synthetic": True},
                    raw={"source": "synthetic"},
                )
            )

    feed = ReplayDataFeed(snapshots)

    data = {
        key: [point.price for point in series]
        for key, series in snapshots.items()
    }
    price_frame = pd.DataFrame(data, index=pd.Index(timeline))

    return feed, price_frame


__all__ = ["build_synthetic_feed"]

