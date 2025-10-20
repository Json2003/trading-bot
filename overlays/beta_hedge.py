"""Helpers for sizing a BTC perpetual hedge to meet a beta target."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _resolve_target(beta: float, target: float | Sequence[float]) -> float:
    """Return the desired beta target, supporting a range specification."""

    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        seq = [float(value) for value in target if value is not None]
        if len(seq) == 2:
            lower, upper = min(seq), max(seq)
            if lower <= beta <= upper:
                return beta
            return upper if beta > upper else lower
    return float(target)


def size_btc_beta_hedge(
    positions: pd.Series,
    betas: pd.Series,
    prices: pd.Series,
    target_beta: float | Sequence[float],
    btc_price: float,
    contract_size: float = 1.0,
) -> float:
    """Return the BTC perpetual contract hedge required to hit ``target_beta``.

    Parameters
    ----------
    positions : pd.Series
        Position sizes in units of the underlying instruments.
    betas : pd.Series
        Beta exposures for each instrument relative to BTC.
    prices : pd.Series
        Latest mark prices for the instruments. Used to compute notionals.
    target_beta : float or Sequence[float]
        Desired portfolio beta after hedging.  A two-value sequence is treated
        as ``(min_beta, max_beta)`` and the hedge will only rebalance when the
        current beta drifts outside this band.
    btc_price : float
        Mark price of the BTC perpetual contract used for hedging.
    contract_size : float, optional
        Contract multiplier. Defaults to 1 which corresponds to a 1 BTC contract.

    Returns
    -------
    float
        Number of BTC perpetual contracts to trade (negative implies shorting).
    """

    positions = positions.astype(float)
    betas = betas.astype(float)
    prices = prices.astype(float)

    beta_index = set(betas.index)
    price_index = set(prices.index)

    notionals: list[float] = []
    beta_contrib: list[float] = []
    for label in positions.index:
        if label not in beta_index or label not in price_index:
            continue
        pos = positions[label]
        beta_val = betas[label]
        price_val = prices[label]
        if pos is None or beta_val is None or price_val is None:
            continue
        if price_val == 0 or price_val != price_val:
            continue
        notional = float(pos) * float(price_val)
        notionals.append(notional)
        beta_contrib.append(notional * float(beta_val))

    if not notionals:
        return 0.0

    portfolio_value = sum(notionals)
    if portfolio_value == 0:
        return 0.0

    portfolio_beta = sum(beta_contrib) / portfolio_value
    target = _resolve_target(portfolio_beta, target_beta)
    beta_gap = portfolio_beta - target
    hedge_notional = -beta_gap * portfolio_value

    if btc_price == 0 or contract_size == 0:
        raise ValueError("btc_price and contract_size must be non-zero")

    contracts = hedge_notional / (btc_price * contract_size)
    if contracts != contracts or contracts in (float("inf"), float("-inf")):
        return 0.0
    return float(contracts)
