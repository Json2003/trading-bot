"""Money engine: position sizing utilities.

Provides conservative, fixed-fractional and risk-per-trade sizing helpers.

Functions
- choose_position_size: returns quantity to buy/sell given account balance and stop-loss
- fixed_fractional: simple fraction of account
- kelly_fraction: returns Kelly fraction suggestion (requires win_rate and win_loss_ratio)

These helpers are intentionally simple and should be adapted to real exchange
contract sizes, minimums, and margin/leverage rules before live use.
"""
from typing import Tuple

def choose_position_size(balance: float, risk_pct: float, entry_price: float, stop_loss_price: float, leverage: float = 1.0, min_qty: float = 0.0) -> Tuple[float, float]:
    """Calculate quantity sized so that potential loss equals balance * risk_pct.

    Args:
        balance: available account balance in quote currency (e.g., USD or USDT)
        risk_pct: fraction of balance to risk per trade (0.01 = 1%)
        entry_price: price at which entry is taken
        stop_loss_price: stop-loss price
        leverage: leverage multiplier (1 = no leverage)
        min_qty: minimum allowed quantity (exchange minimum)

    Returns:
        (qty, notional) where qty is units of base asset (e.g., BTC), notional is qty * entry_price

    Notes:
        - This computes qty = risk_amount / per_unit_risk and then applies leverage by allowing
          larger notional exposure if leverage > 1.
        - Caller must round qty to exchange tick/lot size.
    """
    if balance <= 0 or risk_pct <= 0:
        return 0.0, 0.0

    risk_amount = balance * risk_pct
    per_unit_risk = abs(entry_price - stop_loss_price)
    if per_unit_risk <= 0:
        return 0.0, 0.0

    qty = risk_amount / per_unit_risk
    # apply leverage: with leverage, notional exposure can be larger; quantity remains same but 
    # effective exposure = qty * entry_price * leverage. We return qty unchanged but caller can
    # interpret notional including leverage.
    notional = qty * entry_price
    # enforce minimum
    if qty < min_qty:
        qty = min_qty
        notional = qty * entry_price

    return qty, notional * leverage


def fixed_fractional(balance: float, fraction: float, entry_price: float) -> Tuple[float, float]:
    """Buy a fixed fraction of account notional.

    Returns qty and notional.
    """
    if balance <= 0 or fraction <= 0:
        return 0.0, 0.0
    notional = balance * fraction
    qty = notional / entry_price
    return qty, notional


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Return Kelly fraction given win_rate (0-1) and win_loss_ratio (avg win / avg loss).

    This is a suggestion; in practice use a fraction of Kelly (e.g., 0.25-0.5).
    """
    if win_loss_ratio <= 0:
        return 0.0
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    k = (b * p - q) / b
    if k < 0:
        return 0.0
    return k


def round_qty(qty: float, step: float = 0.0001, min_qty: float = 0.0) -> float:
    """Round quantity to nearest step and enforce minimum quantity.

    - step: smallest tradable lot (e.g., 0.0001 BTC)
    - min_qty: minimum allowed quantity
    """
    if qty <= 0:
        return 0.0
    # round down to be safe
    import math
    steps = math.floor(qty / step)
    q = steps * step
    if q < min_qty:
        return min_qty
    return q


def round_price(price: float, tick: float = 0.01) -> float:
    """Round price to nearest tick (round down)."""
    import math
    return math.floor(price / tick) * tick
