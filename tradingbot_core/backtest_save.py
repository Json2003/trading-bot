"""Helpers for persisting and retrieving backtest results."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json


def _default_serializer(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_backtest_results(
    results: Mapping[str, Any],
    directory: str | Path,
    *,
    prefix: str = "backtest",
    timestamp: datetime | None = None,
) -> Path:
    """Persist a mapping of results to a timestamped JSON file.

    The helper returns the path to the written file which makes it convenient to
    reference in logs or follow-up processing steps.  The directory is created on
    demand if necessary.
    """

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    filename = f"{prefix}_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    target = output_dir / filename

    with target.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, default=_default_serializer, indent=2, sort_keys=True)
        handle.write("\n")

    return target


def load_backtest_results(path: str | Path) -> dict[str, Any]:
    """Load a backtest result file previously written by :func:`save_backtest_results`."""

    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["save_backtest_results", "load_backtest_results"]
