"""Utilities for writing structured result artifacts.

The helpers in this module make it easy for scripts to persist evaluation or
optimization results together with the execution environment metadata.  The
resulting JSON files can later be traced back to the code revision, python
version and dependency set that produced them which greatly simplifies
debugging in CI environments.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def deps_fingerprint() -> str:
    """Return a string fingerprint of the currently installed dependencies.

    The implementation shells out to ``pip freeze`` using the current python
    executable.  Any failure – for instance when ``pip`` is unavailable or the
    command times out – is treated as non-fatal and results in the string
    ``"unavailable"`` which keeps the function side-effect free while still
    signalling that the dependency snapshot could not be collected.
    """

    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            timeout=10,
        )
    except Exception:
        return "unavailable"
    return output.strip()


def save_results(
    path: str | os.PathLike[str],
    results: Mapping[str, Any],
    extra_meta: Mapping[str, Any] | None = None,
) -> None:
    """Persist *results* together with execution metadata to ``path``.

    Parameters
    ----------
    path:
        Location of the JSON file that should be written.  Parent directories
        are created automatically.
    results:
        A mapping containing the result payload.
    extra_meta:
        Optional metadata that should be merged into the automatically gathered
        metadata block.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    meta: MutableMapping[str, Any] = {
        "git_sha": os.getenv("GITHUB_SHA", "<local>"),
        "python": platform.python_version(),
        "deps": deps_fingerprint(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "meta": dict(meta),
        "results": dict(results),
    }

    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


__all__ = ["deps_fingerprint", "save_results"]

