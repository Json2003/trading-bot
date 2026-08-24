#!/usr/bin/env python3
"""Frozen 4-bps funding-positioning variant; research-only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_funding_positioning_reversal as experiment

# New pre-registered variant. The original 8-bps result remains unchanged.
experiment.FUNDING_THRESHOLD = 0.0004

if __name__ == "__main__":
    raise SystemExit(experiment.main())
