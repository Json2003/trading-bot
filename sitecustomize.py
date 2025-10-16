"""Ensure local stub packages shadow binary dependencies.

The kata environment bundles pure-Python compatibility shims for ``numpy`` and
``pandas``.  Some tests manipulate ``sys.path`` to prefer globally installed
packages which would normally pull in the real libraries – these are not
available in the execution environment and would fail to import.  By inserting a
meta path finder we guarantee imports for these packages resolve to the local
stubs regardless of ``sys.path`` ordering.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
from typing import Dict
import sys

_ROOT = Path(__file__).resolve().parent


class _StubFinder(importlib.abc.MetaPathFinder):
    """Intercept imports for stubbed third-party packages."""

    _packages: Dict[str, Path] = {
        "numpy": _ROOT / "numpy",
        "pandas": _ROOT / "pandas",
        "ccxt": _ROOT / "ccxt",
    }

    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        top_level = fullname.split(".", 1)[0]
        package_root = self._packages.get(top_level)
        if package_root is None:
            return None

        relative = fullname.split(".")[1:]
        module_path = package_root.joinpath(*relative)

        if module_path.is_dir():
            init_path = module_path / "__init__.py"
            if not init_path.exists():
                return None
            return importlib.util.spec_from_file_location(
                fullname,
                init_path,
                submodule_search_locations=[str(module_path)],
            )

        py_path = module_path.with_suffix(".py")
        if not py_path.exists():
            return None
        return importlib.util.spec_from_file_location(fullname, py_path)


sys.meta_path.insert(0, _StubFinder())
