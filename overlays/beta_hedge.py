"""Helpers for sizing a BTC perpetual hedge to meet a beta target."""

from __future__ import annotations

import pandas as pd


def size_btc_beta_hedge(
    positions: pd.Series,
    betas: pd.Series,
    prices: pd.Series,
    target_beta: float,
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
    target_beta : float
        Desired portfolio beta after hedging.
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

    df = pd.concat([positions, betas, prices], axis=1, join="inner")
    df.columns = ["position", "beta", "price"]
    if df.empty:
        return 0.0

    notionals = df["position"] * df["price"]
    portfolio_value = notionals.sum()
    if portfolio_value == 0:
        return 0.0

    portfolio_beta = (notionals * df["beta"]).sum() / portfolio_value
    beta_gap = portfolio_beta - target_beta
    hedge_notional = -beta_gap * portfolio_value

    if btc_price == 0 or contract_size == 0:
        raise ValueError("btc_price and contract_size must be non-zero")

    contracts = hedge_notional / (btc_price * contract_size)
    return float(contracts)
