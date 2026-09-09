"""Atomic local archive for a mounted durable research store."""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Any

def archive_segment(source: Path, archive_root: Path, segment_id: str, manifest: dict[str, Any]) -> Path:
    if not segment_id or "/" in segment_id or "\\" in segment_id or segment_id in {".", ".."}:
        raise ValueError("segment_id must be a single safe path component")
    if not source.is_dir():
        raise FileNotFoundError(source)
    target_root = (archive_root / "segments" / segment_id).resolve()
    archive_root_resolved = archive_root.resolve()
    if archive_root_resolved not in target_root.parents:
        raise ValueError("archive target escaped archive root")
    if target_root.exists():
        raise FileExistsError(target_root)
    staging = target_root.parent / (".staging-" + segment_id)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for item in source.iterdir():
            if item.is_symlink() or not item.is_file():
                raise ValueError("segments may contain regular files only")
            shutil.copy2(item, staging / item.name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging, target_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target_root
