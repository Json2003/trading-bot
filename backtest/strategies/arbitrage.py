"""Simple spot-futures arbitrage signal generator.

The module focuses on cross-exchange or spot/futures basis trades that are
popular in liquid crypto markets.  It operates on two price series (spot and
perpetual/futures) and emits position weights for a market-neutral long/short
pair.  The implementation intentionally keeps the interface pandas-friendly so
the signals can be combined with the existing backtest engine or exported to
brokers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class ArbitrageConfig:
    """Configuration for :func:`generate_basis_signals`.

    entry_threshold:
        Minimum basis (futures premium over spot) to enter the trade expressed
        as a fraction, e.g., ``0.002`` for 20 bps.
    exit_threshold:
        Basis level at which the position is closed.  Must be smaller than the
        entry threshold.
    cooldown:
        Optional number of bars to remain flat after closing a position to avoid
        rapid flip-flopping when the basis oscillates around the threshold.
    max_leverage:
        Cap the gross notional allocation of the pair.  Values above one imply
        using borrowed funds (common for futures/spot basis trading).
    lookback:
        Rolling lookback used to compute a z-score of the basis.  Extreme z-score
        readings trigger the same direction as the raw threshold but the z-score
        helps adapt to slow-moving structural changes in the premium.
    z_score_entry:
        Additional entry guard based on the absolute z-score.  Set to ``None`` to
        disable.
    """

    entry_threshold: float = 0.002
    exit_threshold: float = 0.0005
    cooldown: int = 0
    max_leverage: float = 1.0
    lookback: int = 48
    z_score_entry: float | None = 1.5


def _validate_input(df: pd.DataFrame) -> None:
    required = {"timestamp", "spot_close", "futures_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def generate_basis_signals(data: Any, config: ArbitrageConfig | None = None) -> pd.DataFrame:
    """Return long/short allocations for spot vs. futures arbitrage.

    Parameters
    ----------
    data:
        DataFrame-like object containing ``timestamp``, ``spot_close`` and
        ``futures_close`` prices.
    config:
        Optional :class:`ArbitrageConfig`.  Uses the defaults when omitted.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``spot_allocation`` (long when positive),
        ``futures_allocation`` (short when negative) and a ``basis`` helper
        column for diagnostics.  Both allocations are expressed as notional
        fractions of equity.
    """

    cfg = config or ArbitrageConfig()
    df = pd.DataFrame(data).copy()
    _validate_input(df)

    df["basis"] = df["futures_close"] / df["spot_close"] - 1.0
    df["basis_z"] = (
        df["basis"]
        .rolling(int(cfg.lookback), min_periods=int(cfg.lookback))
        .apply(lambda x: 0.0 if np.std(x) == 0 else (x[-1] - np.mean(x)) / np.std(x), raw=True)
    )

    long_spot = (df["basis"] >= cfg.entry_threshold).astype(int)
    if cfg.z_score_entry is not None:
        long_spot &= df["basis_z"].abs() >= cfg.z_score_entry

    flat = (df["basis"] <= cfg.exit_threshold).astype(int)

    spot_alloc = np.zeros(len(df), dtype=float)
    fut_alloc = np.zeros(len(df), dtype=float)

    position = 0
    cooldown_left = 0
    for i in range(len(df)):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if position == 0 and long_spot.iloc[i]:
            position = 1
            spot_alloc[i] = cfg.max_leverage
            fut_alloc[i] = -cfg.max_leverage
        elif position != 0 and flat.iloc[i]:
            position = 0
            cooldown_left = max(cfg.cooldown, 0)

        if position == 1:
            spot_alloc[i] = cfg.max_leverage
            fut_alloc[i] = -cfg.max_leverage

    result = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "basis": df["basis"],
            "spot_allocation": spot_alloc,
            "futures_allocation": fut_alloc,
        }
    )
    return result


def check_live_basis(
    symbol: str = "BTC/USDT",
    threshold: float = 0.005,
    *,
    exchange: Any | None = None,
) -> dict[str, float | int]:
    """Check the basis between spot and futures prices in real time.

    Parameters
    ----------
    symbol:
        Base trading pair to evaluate on the spot market.
    threshold:
        Minimum absolute fractional difference between futures and spot prices
        required to emit a trading signal.
    exchange:
        Optional pre-configured CCXT exchange instance.  When omitted the
        function instantiates :class:`ccxt.binance` lazily.  Passing an
        instance makes the function easier to test and avoids repeated logins
        in production code.

    Returns
    -------
    dict
        Mapping with ``signal`` (``1`` = buy spot / sell futures, ``-1`` = the
        opposite, ``0`` = no trade), ``diff`` (fractional basis), and the
        ``spot``/``futures`` prices that produced the reading.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if exchange is None:
        try:  # pragma: no cover - exercised via dependency injection in tests
            import ccxt  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires missing dep
            raise RuntimeError("ccxt is required to fetch live market data") from exc
        exchange = ccxt.binance()

    spot_ticker: Mapping[str, Any] = exchange.fetch_ticker(symbol)
    futures_symbol = symbol if ":USDT" in symbol else f"{symbol}:USDT"
    futures_ticker: Mapping[str, Any] = exchange.fetch_ticker(futures_symbol)

    spot_price = float(spot_ticker.get("last"))
    futures_price = float(futures_ticker.get("last"))
    if spot_price == 0:
        raise ValueError("Spot price returned by exchange must be non-zero")

    diff = (futures_price - spot_price) / spot_price
    signal = 0
    if abs(diff) > threshold:
        signal = 1 if diff > 0 else -1

    return {
        "signal": signal,
        "diff": diff,
        "spot": spot_price,
        "futures": futures_price,
    }


def generate_threshold_signals(
    data_spot: pd.DataFrame,
    data_futures: pd.DataFrame,
    *,
    threshold: float = 0.005,
    price_column: str = "Close",
) -> pd.DataFrame:
    """Generate spot/futures arbitrage signals on historical data.

    The helper merges two OHLCV-style DataFrames on their ``timestamp`` column,
    computes the fractional basis, and emits long/short instructions whenever
    the absolute basis exceeds ``threshold``.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    required_cols = {"timestamp", price_column}
    missing_spot = required_cols - set(data_spot.columns)
    missing_futures = required_cols - set(data_futures.columns)
    if missing_spot:
        raise ValueError(f"Spot data missing required columns: {sorted(missing_spot)}")
    if missing_futures:
        raise ValueError(f"Futures data missing required columns: {sorted(missing_futures)}")

    spot = pd.DataFrame(data_spot).copy()
    futures = pd.DataFrame(data_futures).copy()

    if len(set(spot["timestamp"])) != len(spot):
        raise ValueError("Spot data contains duplicate timestamps")
    if len(set(futures["timestamp"])) != len(futures):
        raise ValueError("Futures data contains duplicate timestamps")

    spot_records = {
        timestamp: float(price)
        for timestamp, price in zip(spot["timestamp"], spot[price_column])
    }
    futures_records = {
        timestamp: float(price)
        for timestamp, price in zip(futures["timestamp"], futures[price_column])
    }

    common_timestamps = sorted(set(spot_records) & set(futures_records))
    if not common_timestamps:
        return pd.DataFrame(
            columns=["timestamp", "spot_close", "futures_close", "basis", "signal"]
        )

    rows: list[dict[str, Any]] = []
    for ts in common_timestamps:
        spot_close = spot_records[ts]
        futures_close = futures_records[ts]
        if spot_close == 0:
            raise ValueError("Spot prices must be non-zero to compute basis")
        basis = (futures_close - spot_close) / spot_close
        signal = 0
        if basis > threshold:
            signal = 1
        elif basis < -threshold:
            signal = -1
        rows.append(
            {
                "timestamp": ts,
                "spot_close": spot_close,
                "futures_close": futures_close,
                "basis": basis,
                "signal": signal,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["timestamp", "spot_close", "futures_close", "basis", "signal"],
    )


__all__ = [
    "ArbitrageConfig",
    "generate_basis_signals",
    "check_live_basis",
    "generate_threshold_signals",
]
