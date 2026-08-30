#!/usr/bin/env python3
"""Evaluate one frozen one-shot regime-entry hypothesis; research-only."""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

try:
    from scripts.run_cross_asset_regime import (
        BLOCKS, CRYPTO_ASSETS, DISCOVERY_END, END, MACRO_ASSETS, START,
        _summary, _trade, _gate, ema_series, load_close_series,
        load_crypto_bars, regime_at,
    )
except ModuleNotFoundError:
    from run_cross_asset_regime import (
        BLOCKS, CRYPTO_ASSETS, DISCOVERY_END, END, MACRO_ASSETS, START,
        _summary, _trade, _gate, ema_series, load_close_series,
        load_crypto_bars, regime_at,
    )
