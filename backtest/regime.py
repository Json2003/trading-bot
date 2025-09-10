"""Market regime utilities: realized volatility, trend slope, and classification.

Functions:
- realized_vol(close, w): rolling log-return std, annualized-like scaling
- trend_slope(close, w): slow MA slope over ~1/4 of the window
- classify_regime(df, ...): returns a Series of labels

Imports for pandas are guarded to avoid repo-local pandas.py shadows.
"""
from __future__ import annotations

import os, sys
from typing import Any
from dataclasses import dataclass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pd():
    import importlib
    # If a repo-local pandas stub is already imported, remove it so we can load real pandas
    mod = sys.modules.get('pandas')
    if mod is not None:
        mod_file = getattr(mod, '__file__', '') or ''
        try:
            if REPO_ROOT in os.path.abspath(mod_file):
                del sys.modules['pandas']
        except Exception:
            pass
    original = sys.path.copy()
    try:
        repo_paths = {p for p in original if REPO_ROOT in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        return importlib.import_module('pandas')
    finally:
        sys.path = original


def _np():
    import importlib
    return importlib.import_module('numpy')


def realized_vol(close: Any, w: int = 24):
    """Rolling log-return std with simple annualization scaling.

    close: pandas Series of prices
    w: rolling window length (bars)
    """
    pd = _pd(); np = _np()
    r = np.log(close).diff().fillna(0.0)
    return r.rolling(int(w), min_periods=int(w)).std() * (np.sqrt(24 * 365 / float(w)))


def trend_slope(close: Any, w: int = 200):
    """Slow MA slope over roughly a quarter of the window."""
    pd = _pd()
    w = int(w)
    ma = close.rolling(w, min_periods=w).mean()
    return (ma - ma.shift(max(1, w // 4)))


def classify_regime(df: Any,
                    vol_win: int = 48,
                    slope_win: int = 200,
                    vol_low: float = 0.015,
                    vol_high: float = 0.05):
    """Classify each bar into a regime label.

    Returns a pandas Series with values in { 'bull', 'bear', 'chop_lowvol', 'chop_highvol' }.
    """
    pd = _pd()
    cl = df["close"]
    vol = realized_vol(cl, int(vol_win)).ffill()
    slope = trend_slope(cl, int(slope_win)).fillna(0.0)

    reg = pd.Series("chop_lowvol", index=df.index)
    reg[(slope > 0) & (vol >= float(vol_low))] = "bull"
    reg[(slope < 0) & (vol >= float(vol_low))] = "bear"
    reg[(vol > float(vol_high))] = "chop_highvol"  # overrides if very noisy
    return reg


__all__ = ["realized_vol", "trend_slope", "classify_regime"]

# --- Regime parameter helpers -------------------------------------------------

@dataclass
class ATRExit:
    tp_mult: float
    sl_mult: float
    max_bars: int


@dataclass
class TrendATRPolicy:
    # Base crossover + ATR exits + gating
    fast: int
    slow: int
    trend_ma: int
    adx_min: int
    cooldown: int
    atr_pctile_bull: float
    atr_pctile_bear: float
    exits: ATRExit


def params_for_regime(regime: str, base: str = "trend_adx_atr"):
    """
    Returns a tuple (strategy_path, strategy_args dict, exit_args dict) for the regime.

    - strategy_path: module:function for use with dynamic import
    - strategy_args: kwargs for the strategy's generate_signals
    - exit_args: kwargs for ExecConfig related to exits (tp_atr_mult, sl_atr_mult, max_bars)
    """
    if base == "trend_adx_atr":
        # Start from known-good defaults and tweak per regime
        if regime == "bull":
            return (
                "backtest.strategies.trend_adx_atr:generate_signals",
                dict(
                    fast=8,
                    slow=21,
                    trend_ma=200,
                    adx_min=16,
                    atr_pctile_bull=0.30,
                    atr_pctile_bear=0.40,
                    cooldown=3,
                    enable_shorts=True,
                ),
                dict(
                    tp_atr_mult=3.5, sl_atr_mult=1.5, max_bars=16,
                    # New adaptive knobs
                    tp_r_multiple=1.2,
                    trail_atr_mult=1.0,
                ),
            )
        if regime == "bear":
            # Keep shorts on, slightly tighter stops/targets
            return (
                "backtest.strategies.trend_adx_atr:generate_signals",
                dict(
                    fast=8,
                    slow=21,
                    trend_ma=200,
                    adx_min=20,
                    atr_pctile_bull=0.35,
                    atr_pctile_bear=0.50,
                    cooldown=4,
                    enable_shorts=True,
                ),
                dict(
                    tp_atr_mult=3.0, sl_atr_mult=1.25, max_bars=12,
                    # New adaptive knobs: be a touch faster to protect
                    tp_r_multiple=1.0,
                    trail_atr_mult=0.9,
                ),
            )
        if regime == "chop_highvol":
            # Swap to Donchian confirmed to avoid fake crosses
            return (
                "backtest.strategies.donchian_confirmed:generate_signals",
                dict(
                    donchian_n=30,
                    trend_ma=200,
                    adx_min=20,
                    atr_pctile_min=0.35,
                    cooldown=3,
                    enable_shorts=True,
                ),
                dict(
                    tp_atr_mult=2.5, sl_atr_mult=1.25, max_bars=8,
                    # Example: force faster pays in chop
                    tp_r_multiple=1.5,
                    trail_atr_mult=0.8,
                ),
            )
        # chop_lowvol → stand down or be very selective
        return (
            "backtest.strategies.donchian_confirmed:generate_signals",
            dict(
                donchian_n=55,
                trend_ma=200,
                adx_min=18,
                atr_pctile_min=0.25,
                cooldown=4,
                enable_shorts=False,
            ),
            dict(
                tp_atr_mult=2.5, sl_atr_mult=1.25, max_bars=8,
                # Align with chop behavior
                tp_r_multiple=1.5,
                trail_atr_mult=0.8,
            ),
        )

    raise ValueError("Unknown base")


# Export helpers
__all__ = [
    "realized_vol",
    "trend_slope",
    "classify_regime",
    "ATRExit",
    "TrendATRPolicy",
    "params_for_regime",
]


# Small local grid to explore around a given policy/exit setting
import itertools as _it

def around(policy_args: dict, exit_args: dict):
    """Return a small neighborhood grid of (policy_args, exit_args) tuples.

    - Adjusts fast/slow/adx_min/cooldown when present
    - Adjusts Donchian N when present
    - Adjusts ATR exits (tp_atr_mult, sl_atr_mult, max_bars)
    - Returns at most 60 combos
    """
    pa0 = dict(policy_args or {})
    ea0 = dict(exit_args or {})

    fast = [max(5, int(pa0.get("fast", 8)) - 2), int(pa0.get("fast", 8)), int(pa0.get("fast", 8)) + 2] if "fast" in pa0 else [None]
    slow = [max(15, int(pa0.get("slow", 21)) - 2), int(pa0.get("slow", 21)), int(pa0.get("slow", 21)) + 2] if "slow" in pa0 else [None]
    adx  = [max(12, int(pa0.get("adx_min", 18)) - 2), int(pa0.get("adx_min", 18)), int(pa0.get("adx_min", 18)) + 2] if "adx_min" in pa0 else [None]
    cd   = [max(2, int(pa0.get("cooldown", 3)) - 1), int(pa0.get("cooldown", 3)), int(pa0.get("cooldown", 3)) + 1] if "cooldown" in pa0 else [None]

    tp   = [float(ea0.get("tp_atr_mult", 3.0)) - 0.5, float(ea0.get("tp_atr_mult", 3.0)), float(ea0.get("tp_atr_mult", 3.0)) + 0.5]
    sl   = [max(1.0, float(ea0.get("sl_atr_mult", 1.5)) - 0.25), float(ea0.get("sl_atr_mult", 1.5)), float(ea0.get("sl_atr_mult", 1.5)) + 0.25]
    mb   = [max(6, int(ea0.get("max_bars", 12)) - 4), int(ea0.get("max_bars", 12)), int(ea0.get("max_bars", 12)) + 4]
    # New small grids
    tr   = [1.0, 1.2, 1.5]  # tp_r_multiple in R
    pbm  = [0.8, 1.0, 1.2]  # pullback_atr_mult

    dn   = [int(pa0.get("donchian_n", 30)) - 5, int(pa0.get("donchian_n", 30)), int(pa0.get("donchian_n", 30)) + 5] if "donchian_n" in pa0 else [None]

    combos = []
    for f, s, a, c, t, l, m, d, trm, pb in _it.product(fast, slow, adx, cd, tp, sl, mb, dn, tr, pbm):
        pa = dict(pa0)
        if f is not None and "fast" in pa: pa["fast"] = int(f)
        if s is not None and "slow" in pa: pa["slow"] = int(s)
        if a is not None and "adx_min" in pa: pa["adx_min"] = int(a)
        if c is not None and "cooldown" in pa: pa["cooldown"] = int(c)
        if d is not None: pa["donchian_n"] = int(d)
        ea = dict(
            tp_atr_mult=float(t), sl_atr_mult=float(l), max_bars=int(m),
            tp_r_multiple=float(trm), pullback_atr_mult=float(pb)
        )
        combos.append((pa, ea))
    return combos[:60]


# Add around to exports
__all__.append("around")
