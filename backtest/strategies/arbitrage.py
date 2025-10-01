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
from typing import Any

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


__all__ = ["ArbitrageConfig", "generate_basis_signals"]

