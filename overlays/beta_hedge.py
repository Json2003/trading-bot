"""Helpers for sizing a BTC perpetual hedge to meet a beta target."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import math

import pandas as pd


def _normalise_beta_target(target: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
        values = [float(x) for x in target]
        if not values:
            raise ValueError("target_beta sequence cannot be empty")
        return min(values), max(values)
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

    btc_price = float(btc_price)
    contract_size = float(contract_size)
    if btc_price == 0 or contract_size == 0:
        raise ValueError("btc_price and contract_size must be non-zero")

    lower, upper = _normalise_beta_target(target_beta)
    if rebalance_buffer < 0:
        raise ValueError("rebalance_buffer must be non-negative")
    if lower > upper:
        lower, upper = upper, lower

    def _to_mapping(series: pd.Series) -> dict:
        mapping: dict = {}
        index = list(getattr(series, "index", []))
        if index:
            for key in index:
                try:
                    mapping[key] = series[key]
                except Exception:
                    continue
        if not mapping:
            try:
                mapping = dict(series)
            except Exception:
                mapping = {}
        return mapping

    pos_map = _to_mapping(positions)
    beta_map = _to_mapping(betas)
    price_map = _to_mapping(prices)

    aligned_symbols = [
        symbol
        for symbol in pos_map.keys()
        if symbol in beta_map and symbol in price_map
    ]

    notionals: list[float] = []
    beta_values: list[float] = []

    for symbol in aligned_symbols:
        try:
            pos_val = float(pos_map[symbol])
            beta_val = float(beta_map[symbol])
            price_val = float(price_map[symbol])
        except Exception:
            continue
        if not all(math.isfinite(value) for value in (pos_val, beta_val, price_val)):
            continue
        if price_val <= 0:
            continue
        notionals.append(pos_val * price_val)
        beta_values.append(beta_val)

    if not notionals:
        return 0.0

    portfolio_value = float(sum(notionals))
    if abs(portfolio_value) <= 1e-12:
        return 0.0

    beta_contrib = float(sum(notional * beta for notional, beta in zip(notionals, beta_values)))
    portfolio_beta = beta_contrib / portfolio_value

    buffered_lower = lower - rebalance_buffer
    buffered_upper = upper + rebalance_buffer
    if buffered_lower <= portfolio_beta <= buffered_upper:
        return 0.0

    if portfolio_beta > upper:
        target = upper
    elif portfolio_beta < lower:
        target = lower
    else:
        midpoint = (lower + upper) / 2
        target = upper if portfolio_beta >= midpoint else lower

    beta_gap = portfolio_beta - target
    hedge_notional = -beta_gap * portfolio_value

    contracts = hedge_notional / (btc_price * contract_size)
    return float(contracts)
