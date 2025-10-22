"""Utilities to access third-party libraries when local stubs shadow them."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def import_requests():
    """Return the real requests module even if a local stub exists."""
    repo_root = Path(__file__).resolve().parents[1]
    removed: list[tuple[int, str]] = []
    existing = sys.modules.pop("requests", None)
    try:
        for idx in range(len(sys.path) - 1, -1, -1):
            candidate = sys.path[idx]
            try:
                if Path(candidate).resolve() == repo_root:
                    removed.append((idx, candidate))
                    sys.path.pop(idx)
            except Exception:
                continue
        return importlib.import_module("requests")
    except ModuleNotFoundError:
        if existing is not None:
            sys.modules["requests"] = existing
        raise
    finally:
        for idx, value in sorted(removed, key=lambda item: item[0]):
            sys.path.insert(idx, value)
