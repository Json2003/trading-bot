"""Checkpoint and dataset integrity primitives for research workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class CheckpointError(ValueError):
    """Raised when a checkpoint or dataset manifest is unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "last_completed_candle": None,
            "last_evaluated_candle": None,
            "history": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CheckpointError("unsupported checkpoint schema")
    if not isinstance(payload.get("history", []), list):
        raise CheckpointError("checkpoint history must be a list")
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {"dataset_id", "generated_at", "completed_through", "files"}
    missing = required.difference(manifest)
    if missing:
        raise CheckpointError(f"manifest missing fields: {sorted(missing)}")
    if not manifest["dataset_id"] or not manifest["completed_through"]:
        raise CheckpointError("manifest dataset_id and completed_through are required")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise CheckpointError("manifest files must be a non-empty list")
    for item in files:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise CheckpointError("each manifest file needs path and sha256")


def can_advance(previous: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    validate_manifest(manifest)
    prior = previous.get("last_completed_candle")
    current = str(manifest["completed_through"])
    return prior is None or current > str(prior)


def advance_checkpoint(
    previous: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    evaluated_through: str | None = None,
) -> dict[str, Any]:
    if not can_advance(previous, manifest):
        raise CheckpointError("checkpoint cannot move backward or remain unchanged")
    result = dict(previous)
    result["schema_version"] = 1
    result["last_completed_candle"] = manifest["completed_through"]
    if evaluated_through is not None:
        result["last_evaluated_candle"] = evaluated_through
    history = list(result.get("history", []))
    history.append(
        {
            "dataset_id": manifest["dataset_id"],
            "completed_through": manifest["completed_through"],
            "advanced_at": utc_now(),
        }
    )
    result["history"] = history[-1000:]
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
