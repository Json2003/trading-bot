"""Compatibility wrapper exposing ATR indicator in the engine package.

Historically the :class:`ATR` indicator lived under ``engine.atr`` and some
legacy components still import it from there.  The indicator now lives in
``backtest.indicators`` alongside other technical analysis utilities.  This
module re-exports the implementation so existing imports continue to work
without forcing downstream projects to update simultaneously.
"""

from __future__ import annotations

from backtest.indicators import ATR, true_range

__all__ = ["ATR", "true_range"]
