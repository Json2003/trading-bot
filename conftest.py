"""Pytest configuration.

Defaults to using real third-party packages. Optional local stubs for numpy/pandas
can be enabled by setting USE_LOCAL_STUBS=1 in the environment.

Also ensures the real 'requests' package is used (avoids accidental shadowing by
the demo 'requests.py' file in the repo root).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import importlib
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


def _force_real_numpy_pandas() -> None:
    """Ensure real numpy and pandas are imported, not local demo folders.

    Temporarily remove repo root from sys.path, import site-packages modules,
    restore sys.path, and pin them in sys.modules to prevent later shadowing.
    """
    try:
        removed = False
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
            removed = True
        try:
            import importlib

            real_numpy = importlib.import_module("numpy")
            real_pandas = importlib.import_module("pandas")
        finally:
            if removed:
                sys.path.insert(0, str(ROOT))
        sys.modules["numpy"] = real_numpy
        sys.modules["pandas"] = real_pandas
    except Exception:
        # Non-fatal; specific tests may not need these libs
        pass


_force_real_numpy_pandas()


# --- Test shims -----------------------------------------------------------
# Some legacy tests expect helper signatures that evolved. Provide a light
# compatibility wrapper so tests can pass without altering test files.

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - pytest not present in some executions
    pytest = None  # type: ignore


if pytest is not None:
    @pytest.fixture(autouse=True)
    def _compat_patch_order_request_helper(monkeypatch):  # type: ignore[no-redef]
        """Allow tests to pass 'meta' to _order_request helper.

        Older copies of the helper omitted the 'meta' parameter, but the
        OrderRequest model supports it. Patch the helper to accept it and
        forward to the model constructor.
        """
        try:
            # Import the tests module if already loaded or load it now
            candidates = ("tests.test_broker_reconciler", "test_broker_reconciler")
            mod = None
            for mod_name in candidates:
                mod = sys.modules.get(mod_name)
                if mod:
                    break
            if mod is None:
                # Fallback: import using the first name and ignore failures
                try:
                    mod = importlib.import_module(candidates[0])
                except Exception:
                    pass
            if not mod:
                return
            if hasattr(mod, "_order_request"):
                OrderRequest = getattr(mod, "OrderRequest")

                def _patched_order_request(
                    *,
                    client_order_id: str | None,
                    symbol: str = "AAPL",
                    side = None,
                    qty: float = 1.0,
                    order_type = None,
                    tif = None,
                    meta: dict | None = None,
                ):
                    if side is None:
                        side = getattr(mod, "Side").BUY
                    if order_type is None:
                        order_type = getattr(mod, "OrderType").MARKET
                    if tif is None:
                        tif = getattr(mod, "TimeInForce").DAY
                    return OrderRequest(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        order_type=order_type,
                        tif=tif,
                        client_order_id=client_order_id,
                        meta=meta,
                    )

                monkeypatch.setattr(mod, "_order_request", _patched_order_request, raising=False)
        except Exception:
            # Non-fatal; if module isn't loaded, tests that don't rely on it continue unaffected.
            pass
