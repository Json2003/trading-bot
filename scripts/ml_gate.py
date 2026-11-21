#!/usr/bin/env python3
from __future__ import annotations

"""Client-side ML gate helper.

Usage:
    from scripts.ml_gate import infer_gate
    allow, p_up, tag = infer_gate(seq_df.tail(64), url="http://127.0.0.1:8000", threshold=0.55)

Behavior:
    - Extracts raw numeric values (no scaling client-side).
    - Denies on inference failure, invalid probabilities, or model-tag mismatch.
    - A denied failure returns (False, 0.0, None); no model scored that request.
"""

import json
from typing import Optional, Tuple

import numpy as np
import requests


def infer_gate(
    seq_df,
    url: str = "http://127.0.0.1:8000",
    threshold: float = 0.55,
    tag: Optional[str] = None,
    timeout: float = 1.0,
) -> Tuple[bool, float, Optional[str]]:
    try:
        X = np.asarray(seq_df.values, dtype=float)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("seq_df must be a nonempty finite matrix")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between zero and one")
        payload = {"features": X.tolist(), "threshold": float(threshold)}
        if tag:
            payload["tag"] = tag
        r = requests.post(f"{url.rstrip('/')}/infer", json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        probability = float(out["p_up"])
        response_tag = out.get("tag")
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid inference probability")
        if not isinstance(out.get("allow"), bool) or not isinstance(response_tag, str) or not response_tag:
            raise ValueError("missing inference decision or model provenance")
        if tag and tag != response_tag:
            raise ValueError("inference model tag mismatch")
        return out["allow"] and probability >= threshold, probability, response_tag
    except Exception:
        return False, 0.0, None


__all__ = ["infer_gate"]
