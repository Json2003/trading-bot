"""Helpers for sizing a BTC perpetual hedge to meet a beta target."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
        Desired portfolio beta after hedging.  A two-value sequence is treated
        as ``(min_beta, max_beta)`` and the hedge will only rebalance when the
        current beta drifts outside this band.
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

    positions = positions.astype(float)
    betas = betas.astype(float)
    prices = prices.astype(float)

    if hasattr(pd, "concat") and hasattr(positions, "rename"):
        df = pd.concat(
            [
                positions.rename("position"),
                betas.rename("beta"),
                prices.rename("price"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if df.empty:
            return 0.0

        df["notional"] = df["position"] * df["price"]
        portfolio_value = df["notional"].sum()
        if not np.isfinite(portfolio_value) or portfolio_value == 0.0:
            return 0.0

        beta_contrib = (df["notional"] * df["beta"]).sum()
        portfolio_beta = beta_contrib / portfolio_value
    else:
        portfolio_value = 0.0
        beta_contrib = 0.0
        pos_index = list(getattr(positions, "index", range(len(positions))))
        pos_values = list(positions)
        beta_index = list(getattr(betas, "index", range(len(betas))))
        beta_values = list(betas)
        price_index = list(getattr(prices, "index", range(len(prices))))
        price_values = list(prices)

        beta_map = {label: value for label, value in zip(beta_index, beta_values)}
        price_map = {label: value for label, value in zip(price_index, price_values)}

        for idx, label in enumerate(pos_index):
            pos = pos_values[idx] if idx < len(pos_values) else None
            beta_val = beta_map.get(label)
            price_val = price_map.get(label)
            if pos is None or beta_val is None or price_val is None:
                continue
            try:
                notional = float(pos) * float(price_val)
                beta_component = notional * float(beta_val)
            except Exception:
                continue
            if notional != notional:
                continue
            portfolio_value += notional
            beta_contrib += beta_component

        if portfolio_value == 0.0:
            return 0.0

        portfolio_beta = beta_contrib / portfolio_value
    # Determine target handling single value or a range
    if isinstance(target_beta, Sequence) and not isinstance(target_beta, (str, bytes)):
        vals = [float(v) for v in target_beta if v is not None]
        if len(vals) >= 2:
            lower, upper = (min(vals), max(vals))
            buffered_lower = lower - float(rebalance_buffer)
            buffered_upper = upper + float(rebalance_buffer)
            if buffered_lower <= portfolio_beta <= buffered_upper:
                return 0.0
            target = upper if portfolio_beta > upper else lower
        else:
            target = float(vals[0]) if vals else 0.0
    else:
        target = float(target_beta)

    beta_gap = portfolio_beta - target
    hedge_notional = -beta_gap * portfolio_value

    contracts = hedge_notional / (btc_price * contract_size)
    finite = np.isfinite(contracts)
    if not finite:
        return 0.0
    return float(contracts)
