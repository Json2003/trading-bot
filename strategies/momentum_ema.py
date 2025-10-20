"""Dual EMA momentum strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from backtest.indicators import ATR

from .base import Strategy, StrategyContext, StrategySignal


@dataclass
class _EMA:
    span: int
    value: float | None = None

    def update(self, price: float) -> float:
        alpha = 2.0 / (self.span + 1.0)
        if self.value is None:
            self.value = price
        else:
            self.value = alpha * price + (1.0 - alpha) * self.value
        return self.value


class MomentumEMAStrategy(Strategy):
    """Enter long when the fast EMA crosses above the slow EMA and vice versa."""

    def __init__(
        self,
        *,
        symbol: str,
        fast_window: int = 12,
        slow_window: int = 26,
        threshold: float = 0.001,
        session_thresholds: dict[str, float] | None = None,
        atr_period: int = 14,
        atr_pct_threshold: float | None = 0.05,
        order_size: float = 1.0,
        venue: str | None = None,
        market_key: str | None = None,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if atr_pct_threshold is not None and atr_pct_threshold < 0:
            raise ValueError("atr_pct_threshold must be non-negative")
        self.symbol = symbol
        self._fast = _EMA(fast_window)
        self._slow = _EMA(slow_window)
        self._threshold = threshold
        self._default_threshold = threshold
        base_sessions = {"asia": threshold, "europe": threshold, "us": threshold}
        if session_thresholds:
            for key, value in session_thresholds.items():
                base_sessions[key.lower()] = float(value)
        self._session_thresholds = base_sessions
        self._atr_period = int(atr_period)
        self._atr_pct_threshold = float(atr_pct_threshold) if atr_pct_threshold is not None else None
        self._size = order_size
        self._venue = venue
        self._market_key = market_key or (f"{venue}:{symbol}" if venue else symbol)
        self._bias = 0

    def generate_signals(self, context: StrategyContext) -> Iterable[StrategySignal]:
        market = context.data(self._market_key)
        closes: Sequence[float]
        if market.ohlcv:
            closes = [candle.close for candle in market.ohlcv]
        else:
            closes = [market.price]

        for price in closes:
            fast = self._fast.update(price)
            slow = self._slow.update(price)

        threshold = self._session_thresholds.get((market.session or "").lower(), self._default_threshold)
        if self._atr_pct_threshold is not None:
            atr_pct = self._atr_percent(market)
            if atr_pct is not None and atr_pct > self._atr_pct_threshold:
                return []

        bias = 0
        if fast - slow > threshold * market.price:
            bias = 1
        elif slow - fast > threshold * market.price:
            bias = -1

        if bias == self._bias:
            return []

        self._bias = bias
        if bias > 0:
            return [
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="buy",
                    quantity=self._size,
                    price=market.price,
                    venue=self._venue,
                    tags={
                        "type": "momentum",
                        "direction": "long",
                        "market_key": self._market_key,
                    },
                )
            ]
        elif bias < 0:
            return [
                StrategySignal(
                    strategy=context.strategy,
                    symbol=self.symbol,
                    side="sell",
                    quantity=self._size,
                    price=market.price,
                    venue=self._venue,
                    tags={
                        "type": "momentum",
                        "direction": "short",
                        "market_key": self._market_key,
                    },
                )
            ]
        return []

    def _atr_percent(self, market) -> float | None:
        candles = market.ohlcv
        if not candles or market.price <= 0:
            return None
        indicator = ATR(window=self._atr_period)
        sample = candles[-(self._atr_period + 1) :]
        for candle in sample:
            indicator.update(candle)
        value = indicator.value
        if value is None:
            return None
        return value / market.price


__all__ = ["MomentumEMAStrategy"]
