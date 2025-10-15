"""Local pandas shim: prefer real pandas from site-packages, otherwise use package stub.
This avoids accidental shadowing while providing a minimal fallback for tests.
"""
"""Local pandas shim: prefer real pandas from site-packages, otherwise use
package-local stub. This avoids accidental shadowing while providing a
minimal fallback for tests and CI.

The loader temporarily removes any sys.path entries that resolve to the
project root (including the empty string entry that Python inserts when the
current working directory is the project root) so that `import pandas` will
resolve to the site-packages installation instead of importing this shim.
"""
import importlib
import sys
from pathlib import Path


def _load_real_pandas():
    """Try to import the real pandas from site-packages.

    This will temporarily remove any sys.path entries that point to the
    repository root (or '' when cwd == repo root). Removed entries are
    restored before returning.
    """
    project_root = Path(__file__).resolve().parent
    removed = []
    try:
        # Identify entries that resolve to the project root (or empty entry
        # that implies the cwd). We operate on a copy of sys.path to avoid
        # mutating while iterating.
        for p in list(sys.path):
            try:
                if p == "":
                    # '' means import from cwd; treat it as project root if cwd matches
                    if Path.cwd().resolve() == project_root:
                        sys.path.remove(p)
                        removed.append(p)
                else:
                    try:
                        if Path(p).resolve() == project_root:
                            sys.path.remove(p)
                            removed.append(p)
                    except Exception:
                        # ignore entries that can't be resolved
                        continue
            except ValueError:
                # already removed by another thread/process; ignore
                continue

        # Also protect against a stale entry in sys.modules that points to
        # this shim. If so, remove it so importlib will load the real package.
        cur = sys.modules.get("pandas")
        if cur is not None:
            cur_file = getattr(cur, "__file__", None)
            if cur_file:
                try:
                    if Path(cur_file).resolve() == Path(__file__).resolve():
                        sys.modules.pop("pandas", None)
                except Exception:
                    pass

        return importlib.import_module("pandas")
    except Exception:
        return None
    finally:
        # restore removed entries in reverse order
        for p in reversed(removed):
            sys.path.insert(0, p)


# attempt to load real pandas first
_real = _load_real_pandas()
if _real is not None:
    pd = _real
    # re-export common names for `from pandas import ...` cases
    globals().update({k: getattr(pd, k) for k in dir(pd)})
else:
    # fallback to the package-local stub
    from tradingbot_ibkr import _pandas_stub as pd
    globals().update({k: getattr(pd, k) for k in dir(pd) if not k.startswith("_")})


# make `pd` available for modules that do `import pandas as pd`
__all__ = ["pd"]

