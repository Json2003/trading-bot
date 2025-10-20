"""Helpers for sizing a BTC perpetual hedge to meet a beta target."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd


def _normalise_beta_target(target: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
        values = [float(x) for x in target]
        if not values:
            raise ValueError("target_beta sequence cannot be empty")
        lower = min(values)
        upper = max(values)
        return lower, upper
    target_val = float(target)
    return target_val, target_val


def size_btc_beta_hedge(
    positions: pd.Series,
    betas: pd.Series,
    prices: pd.Series,
    target_beta: float | Sequence[float],
    btc_price: float,
    contract_size: float = 1.0,
    rebalance_buffer: float = 0.0,
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
        Desired portfolio beta after hedging. A sequence defines the acceptable
        beta band ``(lower, upper)``.
    btc_price : float
        Mark price of the BTC perpetual contract used for hedging.
    contract_size : float, optional
        Contract multiplier. Defaults to 1 which corresponds to a 1 BTC contract.
    rebalance_buffer : float, optional
        Additional tolerance added to the beta band before trading.

    Returns
    -------
    float
        Number of BTC perpetual contracts to trade (negative implies shorting).
    """
    if btc_price == 0 or contract_size == 0:
        raise ValueError("btc_price and contract_size must be non-zero")

    lower, upper = _normalise_beta_target(target_beta)
    if rebalance_buffer < 0:
        raise ValueError("rebalance_buffer must be non-negative")

    aligned_symbols = []
    position_values: list[float] = []
    beta_values: list[float] = []
    price_values: list[float] = []

    for symbol in positions.index:
        if symbol not in betas.index or symbol not in prices.index:
            continue
        try:
            pos_val = float(positions[symbol])
            beta_val = float(betas[symbol])
            price_val = float(prices[symbol])
        except Exception:
            continue
        if not math.isfinite(pos_val) or not math.isfinite(beta_val) or not math.isfinite(price_val):
            continue
        aligned_symbols.append(symbol)
        position_values.append(pos_val)
        beta_values.append(beta_val)
        price_values.append(price_val)

    if not aligned_symbols:
        return 0.0

    notionals = [pos * price for pos, price in zip(position_values, price_values)]
    portfolio_value = sum(notionals)
    if portfolio_value == 0:
        return 0.0

    beta_contrib = sum(notional * beta for notional, beta in zip(notionals, beta_values))
    portfolio_beta = beta_contrib / portfolio_value

    if lower > upper:
        lower, upper = upper, lower

    buffered_lower = lower - rebalance_buffer
    buffered_upper = upper + rebalance_buffer

    if buffered_lower <= portfolio_beta <= buffered_upper:
        return 0.0

    if portfolio_beta > upper:
        target = upper
    elif portfolio_beta < lower:
        target = lower
    else:
        # Already inside the original band but outside the buffered one -> nudge to nearest edge.
        target = upper if portfolio_beta > (lower + upper) / 2 else lower

    beta_gap = portfolio_beta - target
    hedge_notional = -beta_gap * portfolio_value

    contracts = hedge_notional / (btc_price * contract_size)
    return float(contracts)
