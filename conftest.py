"""Pytest configuration ensuring local stub packages shadow binary wheels."""

from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
from typing import Dict
import sys

ROOT = Path(__file__).resolve().parent


class _StubFinder(importlib.abc.MetaPathFinder):
    """Intercept imports for packages that have local pure-Python shims."""

    packages: Dict[str, Path] = {
        "numpy": ROOT / "numpy",
        "pandas": ROOT / "pandas",
    }

    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        top = fullname.split(".", 1)[0]
        package_root = self.packages.get(top)
        if package_root is None:
            return None

        parts = fullname.split(".")[1:]
        module_path = package_root.joinpath(*parts)

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


# Insert at the front so it takes precedence over the default path-based finder
sys.meta_path.insert(0, _StubFinder())
