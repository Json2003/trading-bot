#!/usr/bin/env python3
from __future__ import annotations

"""Client-side ML gate helper.

Usage:
    from scripts.ml_gate import infer_gate
    allow, p_up, tag = infer_gate(seq_df.tail(64), url="http://127.0.0.1:8000", threshold=0.55)

Behavior:
    - Extracts raw numeric values (no scaling client-side).
    - Falls back to allow=True on any failure so trading isn’t blocked.
"""

import json
from typing import Optional, Tuple

import numpy as np
import requests


def infer_gate(seq_df, url: str = "http://127.0.0.1:8000", threshold: float = 0.55, tag: Optional[str] = None,
               timeout: float = 1.0) -> Tuple[bool, float, Optional[str]]:
    try:
        X = np.asarray(seq_df.values, dtype=float)
        if X.ndim != 2:
            raise ValueError("seq_df must be 2D")
        payload = {"features": X.tolist(), "threshold": float(threshold)}
        if tag:
            payload["tag"] = tag
        r = requests.post(f"{url.rstrip('/')}/infer", json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        return bool(out.get("allow", True)), float(out.get("p_up", 0.0)), out.get("tag")
    except Exception:
        # Fail-open
        return True, 0.5, tag


__all__ = ["infer_gate"]

