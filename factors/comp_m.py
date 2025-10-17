"""Cross-sectional momentum factor (Comp-M) implementation."""

from __future__ import annotations

import pandas as pd


def compute_comp_m_factor(
    prices: pd.DataFrame,
    lookback: int,
    lag: int = 1,
    neutralize: bool = True,
) -> pd.DataFrame:
    """Return the Comp-M cross-sectional momentum ranks.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price table indexed by timestamp with tickers along the columns.
    lookback : int
        Number of periods used for the momentum window.
    lag : int, optional
        How many most recent bars to skip to avoid look-ahead bias. Defaults to 1.
    neutralize : bool, optional
        If ``True``, demean ranks each period to keep the factor dollar-neutral.

    Returns
    -------
    pd.DataFrame
        Cross-sectional ranks scaled to the interval [-1, 1].
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if prices.empty:
        return prices.copy()

    momentum = prices.pct_change(periods=lookback)
    if lag:
        momentum = momentum.shift(lag)

    ranks = momentum.rank(axis=1, pct=True, method="average") * 2 - 1
    if neutralize:
        ranks = ranks.sub(ranks.mean(axis=1), axis=0)
    return ranks.dropna(how="all")
