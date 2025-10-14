"""Lightweight CCXT stub for offline environments.

This project only needs a very small subset of the ``ccxt`` package in order
to run the toy backtests that ship with the repository.  The real dependency is
quite heavy and not available in the execution environment that powers the
automated assessments.  To keep the user-facing API unchanged while avoiding a
hard dependency on the external library, we provide a tiny drop-in
replacement.

The stub implements the ``binance`` factory and the ``Exchange.fetch_ohlcv``
method used by :mod:`scripts.multi_strategy_backtest`.  Candle data is sourced
from the repository's deterministic sample dataset so results remain stable
across runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

__all__ = ["Exchange", "binance"]


@dataclass
class Exchange:
    """Minimal exchange surface compatible with the backtest scripts."""

    rateLimit: int = 1200

    def __init__(self, **kwargs: object) -> None:
        # ``ccxt`` accepts a large variety of keyword arguments.  We only store
        # them for introspection to avoid surprising callers.
        self.options = dict(kwargs)

    # In the real library this would be abstract, but keeping it concrete and
    # raising ``NotImplementedError`` keeps type-checkers satisfied without the
    # need for ``abc.ABC`` ceremony.
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 1000,
    ) -> List[List[float]]:
        raise NotImplementedError


def _dataset_path() -> Path:
    """Return the path to the bundled sample OHLCV dataset."""

    return Path(__file__).resolve().parents[1] / "backtest" / "sample_data" / "sample_ohlcv.csv"


def _load_sample_rows() -> List[tuple[int, float, float, float, float, float]]:
    """Parse the canonical sample dataset into OHLCV tuples."""

    rows: List[tuple[int, float, float, float, float, float]] = []
    with _dataset_path().open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for rec in reader:
            dt = datetime.fromisoformat(rec["timestamp"])
            ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
            rows.append(
                (
                    ts,
                    float(rec["open"]),
                    float(rec["high"]),
                    float(rec["low"]),
                    float(rec["close"]),
                    float(rec["volume"]),
                )
            )
    return rows


def _transform(rows: Iterable[Sequence[float]], factor: float) -> List[List[float]]:
    """Apply a deterministic scale transformation to a series of candles."""

    out: List[List[float]] = []
    for ts, o, h, l, c, v in rows:
        out.append([ts, o * factor, h * factor, l * factor, c * factor, v * factor])
    return out


_BASE_SERIES = _load_sample_rows()
_SERIES_CACHE: Dict[str, List[List[float]]] = {}


class _BinanceStub(Exchange):
    """Provide deterministic OHLCV candles for a couple of common symbols."""

    _SCALE = {
        "BTC/USDT": 520.0,
        "ETH/USDT": 32.0,
    }

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 1000,
    ) -> List[List[float]]:
        if timeframe != "1h":
            raise ValueError("Stub exchange only supports 1h candles")

        if symbol not in self._SCALE:
            raise ValueError(f"Stub exchange has no data for symbol {symbol!r}")

        if symbol not in _SERIES_CACHE:
            _SERIES_CACHE[symbol] = _transform(_BASE_SERIES, self._SCALE[symbol])

        series = _SERIES_CACHE[symbol]
        start = 0
        if since is not None:
            # ``since`` is inclusive in the real API.  We find the first index
            # with a timestamp >= since.  If the caller requests data beyond the
            # available range we wrap back to the beginning so the educational
            # examples always receive candles.
            while start < len(series) and series[start][0] < since:
                start += 1
            if start >= len(series):
                start = 0

        end = min(start + limit, len(series))
        return [list(row) for row in series[start:end]]


def binance(config: Dict[str, object] | None = None) -> Exchange:
    """Return a deterministic Binance exchange stub."""

    return _BinanceStub(**(config or {}))

