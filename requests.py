"""Local shim: import the real requests package from site-packages to avoid
shadowing. This file previously contained a static stub which broke code that
imports submodules (requests.adapters)."""
import importlib.util, sys, os

def _load_real_requests():
    for p in list(sys.path):
        if not p:
            continue
        candidate = os.path.join(p, 'requests', '__init__.py')
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location('requests', candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules['requests'] = mod
                spec.loader.exec_module(mod)
                return mod
    # final fallback to normal import (will raise)
    return importlib.import_module('requests')

_requests = _load_real_requests()
globals().update({k: getattr(_requests, k) for k in dir(_requests) if not k.startswith('__')})
