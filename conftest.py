"""Pytest configuration.

Defaults to using real third-party packages. Optional local stubs for numpy/pandas
can be enabled by setting USE_LOCAL_STUBS=1 in the environment.

Also ensures the real 'requests' package is used (avoids accidental shadowing by
the demo 'requests.py' file in the repo root).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
import os
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


def _maybe_enable_local_stubs() -> None:
    """Enable local stubs finder only if explicitly requested via env var.

    Set USE_LOCAL_STUBS=1 to route numpy/pandas imports to local pure-Python shims.
    """
    use_stubs = os.getenv("USE_LOCAL_STUBS", "0") in ("1", "true", "yes")
    if use_stubs:
        # Insert at the front so it takes precedence over the default path-based finder
        sys.meta_path.insert(0, _StubFinder())


def _force_real_requests() -> None:
    """Ensure site-packages 'requests' is imported instead of local demo module.

    Temporarily remove the repository root from sys.path to import the real package,
    then restore path ordering and pin the loaded module in sys.modules.
    """
    try:
        removed = False
        if str(ROOT) in sys.path:
            # Remove repo root to avoid importing the local demo file
            sys.path.remove(str(ROOT))
            removed = True
        try:
            import importlib

            real_requests = importlib.import_module("requests")
        except Exception:
            # Fallback to pip vendored requests if standard import fails
            import importlib

            real_requests = importlib.import_module("pip._vendor.requests")
        finally:
            if removed:
                sys.path.insert(0, str(ROOT))

        sys.modules["requests"] = real_requests
    except Exception:
        # Non-fatal; tests that don't use requests won't be affected
        pass


_maybe_enable_local_stubs()
_force_real_requests()
