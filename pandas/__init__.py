"""Prefer the installed pandas package, with a pure-Python fallback for tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_installed_pandas():
    repo_root = Path(__file__).resolve().parent.parent
    original_path = sys.path.copy()
    existing = sys.modules.pop(__name__, None)
    try:
        sys.path = [entry for entry in original_path if Path(entry or ".").resolve() != repo_root]
        return importlib.import_module(__name__)
    except (ImportError, ModuleNotFoundError):
        return None
    finally:
        sys.path = original_path
        if existing is not None and __name__ not in sys.modules:
            sys.modules[__name__] = existing


_installed = _load_installed_pandas()
if _installed is None:
    from ._stub import *  # noqa: F401,F403
else:
    globals().update(_installed.__dict__)
    sys.modules[__name__] = _installed
