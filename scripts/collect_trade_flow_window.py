#!/usr/bin/env python3
"""Collect a durable, restart-safe public Binance trade-flow research window.

This controller runs bounded observer segments, stores each segment under a
unique immutable ID, and advances a timestamped checkpoint only after the
segment is finalized. GCS is the durable archive for ephemeral CI runners;
a persistent local state directory is also supported for self-hosted runs.

The controller has no account, broker, order, leverage, or promotion path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from monitor_binance_trade_flow import run_monitor

UTC = timezone.utc
SCHEMA_VERSION = 1
WINDOW_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_bucket(value: str) -> datetime:
    return _parse_iso(value)


def _compact_timestamp(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)[:14]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_window_id(window_id: str) -> None:
    if not WINDOW_ID_RE.fullmatch(window_id):
        raise ValueError(
            "window_id must contain only letters, numbers, '.', '_' or '-' "
            "and must be at most 81 characters"
        )


def _segment_number(segment_id: str) -> int:
    match = re.fullmatch(r"segment-(\d{6})", segment_id)
    if not match:
        raise ValueError(f"invalid segment ID: {segment_id}")
    return int(match.group(1))


def _segment_manifests(state_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((state_dir / "segments").glob("segment-*/segment_manifest.json")):
        payload = _load_json(path)
        if payload is not None:
            payload["_path"] = str(path)
            manifests.append(payload)
    manifests.sort(key=lambda item: _segment_number(str(item["segment_id"])))
    return manifests


def _initial_cursor(window_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "window_id": window_manifest["window_id"],
        "next_segment_number": 1,
        "segments_completed": 0,
        "normalized_event_count": 0,
        "completed_summary_row_count": 0,
        "gap_count": 0,
        "overlap_count": 0,
        "data_through": window_manifest["start_checkpoint_utc"],
        "last_segment_id": None,
        "updated_at": window_manifest["start_checkpoint_utc"],
    }


def _checkpoint_after(
    previous: dict[str, Any], segment: dict[str, Any]
) -> dict[str, Any]:
    status = segment["continuity_status"]
    return {
        "schema_version": SCHEMA_VERSION,
        "window_id": previous["window_id"],
        "next_segment_number": _segment_number(segment["segment_id"]) + 1,
        "segments_completed": int(previous["segments_completed"]) + 1,
        "normalized_event_count": int(previous["normalized_event_count"])
        + int(segment["normalized_event_count"]),
        "completed_summary_row_count": int(previous["completed_summary_row_count"])
        + int(segment["completed_summary_row_count"]),
        "gap_count": int(previous["gap_count"]) + int(status == "gap"),
        "overlap_count": int(previous["overlap_count"]) + int(status == "overlap"),
        "data_through": segment["observed_end_exclusive_utc"],
        "last_segment_id": segment["segment_id"],
        "updated_at": segment["finished_at_utc"],
    }


def _recover_cursor(
    checkpoint: dict[str, Any] | None,
    window_manifest: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    cursor = checkpoint or _initial_cursor(window_manifest)
    expected = int(cursor["next_segment_number"])
    for segment in manifests:
        number = _segment_number(str(segment["segment_id"]))
        if number < expected:
            continue
        if number != expected:
            break
        cursor = _checkpoint_after(cursor, segment)
        expected += 1
    return cursor


def _initialize_window_manifest(
    state_dir: Path,
    window_id: str,
    symbols: list[str],
    large_trade_notional: float,
    target_days: float,
) -> tuple[dict[str, Any], bool]:
    path = state_dir / "window_manifest.json"
    existing = _load_json(path)
    if existing is not None:
        expected = {
            "window_id": window_id,
            "symbols": sorted(symbols),
            "large_trade_notional": float(large_trade_notional),
            "target_days": float(target_days),
        }
        actual = {
            "window_id": existing.get("window_id"),
            "symbols": sorted(existing.get("symbols", [])),
            "large_trade_notional": float(existing.get("large_trade_notional", -1)),
            "target_days": float(existing.get("target_days", -1)),
        }
        if actual != expected:
            raise ValueError(
                "existing window_manifest.json does not match the frozen "
                "window configuration; use a new window_id"
            )
        return existing, False

    started = _now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "window_id": window_id,
        "symbols": sorted(symbols),
        "large_trade_notional": float(large_trade_notional),
        "target_days": float(target_days),
        "start_checkpoint_utc": _iso(started),
        "planned_end_checkpoint_utc": _iso(
            started + timedelta(days=target_days)
        ),
        "created_at_utc": _iso(started),
        "source": "Binance USD-M public WebSocket market streams",
        "streams": [
            "symbol@aggTrade",
            "symbol@bookTicker",
            "!forceOrder@arr",
        ],
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "frozen_configuration": True,
        "unseen_data_checkpoint_policy": (
            "Every completed segment creates an immutable timestamped "
            "checkpoint. Analyses must pin a checkpoint before reading newer "
            "segments; rows newer than that checkpoint remain untouched."
        ),
        "continuity_policy": (
            "A segment is not represented as continuous when its observed "
            "minute coverage has a gap or overlap. Gaps are recorded explicitly."
        ),
    }
    _atomic_write_json(path, manifest)
    return manifest, True


def _read_summary_bounds(
    summary_path: Path,
) -> tuple[str | None, str | None, int]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None, 0
    all_buckets = [row["bucket"] for row in rows if row.get("bucket")]
    completed_rows = [
        row
        for row in rows
        if str(row.get("completed", "")).lower() == "true"
    ]
    completed_buckets = [
        row["bucket"] for row in completed_rows if row.get("bucket")
    ]
    first_bucket = min(all_buckets) if all_buckets else None
    last_completed = max(completed_buckets) if completed_buckets else None
    return first_bucket, last_completed, len(completed_rows)


def _build_segment_manifest(
    segment_id: str,
    window_manifest: dict[str, Any],
    monitor_manifest: dict[str, Any],
    segment_dir: Path,
    previous_cursor: dict[str, Any],
    max_gap_seconds: float,
) -> dict[str, Any]:
    summary_path = segment_dir / "completed_minute_flow.csv"
    first_bucket, last_completed_bucket, completed_count = _read_summary_bounds(
        summary_path
    )
    if first_bucket is None:
        raise ValueError("segment produced no minute summaries")
    if last_completed_bucket is None:
        raise ValueError("segment produced no completed minute summary")
    observed_end = _parse_bucket(last_completed_bucket) + timedelta(minutes=1)
    previous_end = _parse_iso(str(previous_cursor["data_through"]))
    first_observed = _parse_bucket(first_bucket)
    delta_seconds = (first_observed - previous_end).total_seconds()
    if int(previous_cursor["segments_completed"]) == 0:
        continuity_status = "initial"
        gap_seconds = 0.0
        overlap_seconds = 0.0
    elif delta_seconds > max_gap_seconds:
        continuity_status = "gap"
        gap_seconds = delta_seconds
        overlap_seconds = 0.0
    elif delta_seconds < 0:
        continuity_status = "overlap"
        gap_seconds = 0.0
        overlap_seconds = -delta_seconds
    else:
        continuity_status = "continuous"
        gap_seconds = max(0.0, delta_seconds)
        overlap_seconds = 0.0

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(segment_dir.iterdir()):
        if path.is_file() and path.name != "segment_manifest.json":
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "window_id": window_manifest["window_id"],
        "segment_id": segment_id,
        "started_at_utc": monitor_manifest["started_at"],
        "finished_at_utc": monitor_manifest["finished_at"],
        "duration_seconds": monitor_manifest["duration_seconds"],
        "first_observed_bucket_utc": first_bucket,
        "last_completed_bucket_utc": last_completed_bucket,
        "observed_end_exclusive_utc": _iso(observed_end),
        "gap_seconds_from_previous": gap_seconds,
        "overlap_seconds_from_previous": overlap_seconds,
        "continuity_status": continuity_status,
        "normalized_event_count": monitor_manifest["normalized_event_count"],
        "completed_summary_row_count": completed_count,
        "connected": monitor_manifest["connected"],
        "errors": monitor_manifest["errors"],
        "files": files,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "promotion_allowed": False,
    }


def _checkpoint_record(
    window_manifest: dict[str, Any],
    cursor: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_id = (
        f"{window_manifest['window_id']}-"
        f"{segment['segment_id']}-"
        f"{_compact_timestamp(segment['observed_end_exclusive_utc'])}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "checkpoint_type": "unseen-data",
        "window_id": window_manifest["window_id"],
        "created_at_utc": _iso(_now()),
        "data_start_utc": window_manifest["start_checkpoint_utc"],
        "data_through_utc": segment["observed_end_exclusive_utc"],
        "inclusive_segment_id": segment["segment_id"],
        "segments_completed": cursor["segments_completed"],
        "continuity_status": (
            "continuous"
            if cursor["gap_count"] == 0 and cursor["overlap_count"] == 0
            else "contains-explicit-gaps-or-overlaps"
        ),
        "normalized_event_count": cursor["normalized_event_count"],
        "completed_summary_row_count": cursor["completed_summary_row_count"],
        "unseen_data_rule": (
            "Do not use segments newer than data_through_utc during "
            "discovery or tuning for analyses pinned to this checkpoint."
        ),
        "research_only": True,
        "orders_allowed": False,
        "promotion_allowed": False,
    }


class GCSArchive:
    """Durable GCS archive with control-file recovery for ephemeral runners."""

    def __init__(self, bucket_name: str, prefix: str) -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-storage is required when --gcs-bucket is used"
            ) from exc
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.strip("/")

    def _name(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def sync_controls(self, state_dir: Path) -> None:
        prefix = f"{self.prefix}/" if self.prefix else ""
        control_files = {
            "window_manifest.json",
            "checkpoint.json",
        }
        for blob in self.bucket.list_blobs(prefix=prefix):
            relative = blob.name[len(prefix) :]
            if not (
                relative in control_files
                or relative.startswith("checkpoints/")
                or relative.endswith("/segment_manifest.json")
            ):
                continue
            target = state_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=f".{target.name}.", text=False
            )
            os.close(fd)
            try:
                blob.download_to_filename(temporary_name)
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            if relative == "checkpoint.json":
                _atomic_write_text(
                    state_dir / ".checkpoint.generation",
                    str(blob.generation),
                )

    def create_json(self, relative: str, payload: dict[str, Any]) -> None:
        blob = self.bucket.blob(self._name(relative))
        blob.upload_from_string(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            content_type="application/json",
            if_generation_match=0,
        )

    def update_checkpoint(
        self,
        payload: dict[str, Any],
        expected_generation: int | None,
    ) -> int:
        blob = self.bucket.blob(self._name("checkpoint.json"))
        kwargs: dict[str, Any] = {}
        if expected_generation is None:
            kwargs["if_generation_match"] = 0
        else:
            kwargs["if_generation_match"] = expected_generation
        blob.upload_from_string(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            content_type="application/json",
            **kwargs,
        )
        return int(blob.generation)

    def upload_segment(self, segment_dir: Path) -> None:
        files = sorted(path for path in segment_dir.iterdir() if path.is_file())
        manifest = [path for path in files if path.name == "segment_manifest.json"]
        for path in [path for path in files if path.name != "segment_manifest.json"]:
            relative = f"segments/{segment_dir.name}/{path.name}"
            self.bucket.blob(self._name(relative)).upload_from_filename(
                str(path),
                content_type="application/gzip"
                if path.suffix == ".gz"
                else "application/octet-stream",
            )
        for path in manifest:
            relative = f"segments/{segment_dir.name}/{path.name}"
            self.bucket.blob(self._name(relative)).upload_from_filename(
                str(path), content_type="application/json"
            )


def _recover_staging(state_dir: Path) -> None:
    staging = state_dir / "staging"
    if not staging.exists():
        return
    for path in sorted(staging.glob("segment-*")):
        recovery = state_dir / "recovery" / f"{path.name}-{_compact_timestamp(_iso(_now()))}"
        recovery.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(recovery))
        _atomic_write_json(
            recovery / "recovery_manifest.json",
            {
                "status": "abandoned-before-segment-finalization",
                "recovered_at_utc": _iso(_now()),
                "research_only": True,
                "orders_allowed": False,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gcs-bucket", default=None)
    parser.add_argument("--gcs-prefix", default="research/binance-trade-flow")
    parser.add_argument("--target-days", type=float, default=90.0)
    parser.add_argument("--segment-seconds", type=float, default=3300.0)
    parser.add_argument("--large-trade-notional", type=float, default=100_000.0)
    parser.add_argument("--reconnect-attempts", type=int, default=6)
    parser.add_argument("--max-gap-seconds", type=float, default=120.0)
    parser.add_argument("--fail-on-gap", action="store_true")
    args = parser.parse_args()

    _validate_window_id(args.window_id)
    symbols = sorted({symbol.upper() for symbol in args.symbols})
    if not symbols:
        raise ValueError("at least one symbol is required")
    if args.target_days <= 0:
        raise ValueError("target-days must be positive")
    if args.segment_seconds <= 0 or args.segment_seconds > 3600:
        raise ValueError("segment-seconds must be between 1 and 3600")
    if args.large_trade_notional <= 0:
        raise ValueError("large-trade-notional must be positive")
    if args.max_gap_seconds < 0:
        raise ValueError("max-gap-seconds cannot be negative")

    state_dir = args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    archive = (
        GCSArchive(args.gcs_bucket, args.gcs_prefix)
        if args.gcs_bucket
        else None
    )
    if archive is not None:
        archive.sync_controls(state_dir)
    _recover_staging(state_dir)

    window_manifest, created = _initialize_window_manifest(
        state_dir,
        args.window_id,
        symbols,
        args.large_trade_notional,
        args.target_days,
    )
    if archive is not None and created:
        try:
            archive.create_json("window_manifest.json", window_manifest)
        except Exception:
            archive.sync_controls(state_dir)
            window_manifest = _load_json(state_dir / "window_manifest.json")
            if window_manifest is None:
                raise

    checkpoint = _load_json(state_dir / "checkpoint.json")
    manifests = _segment_manifests(state_dir)
    cursor = _recover_cursor(checkpoint, window_manifest, manifests)
    planned_end = _parse_iso(window_manifest["planned_end_checkpoint_utc"])
    remaining = (planned_end - _now()).total_seconds()
    if remaining <= 0:
        completion = {
            "schema_version": SCHEMA_VERSION,
            "window_id": args.window_id,
            "status": "complete",
            "completed_at_utc": _iso(_now()),
            "data_start_utc": window_manifest["start_checkpoint_utc"],
            "data_through_utc": cursor["data_through"],
            "segments_completed": cursor["segments_completed"],
            "gap_count": cursor["gap_count"],
            "overlap_count": cursor["overlap_count"],
            "research_only": True,
            "orders_allowed": False,
            "promotion_allowed": False,
        }
        completion_path = state_dir / "window_complete.json"
        if not completion_path.exists():
            _atomic_write_json(completion_path, completion)
            if archive is not None:
                archive.create_json("window_complete.json", completion)
        print(json.dumps(completion, indent=2))
        return 0

    segment_number = int(cursor["next_segment_number"])
    segment_id = f"segment-{segment_number:06d}"
    staging_dir = state_dir / "staging" / segment_id
    final_dir = state_dir / "segments" / segment_id
    if final_dir.exists():
        raise RuntimeError(f"refusing to overwrite finalized segment {segment_id}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    duration = min(args.segment_seconds, remaining)
    monitor_manifest = run_monitor(
        symbols,
        staging_dir,
        duration,
        args.large_trade_notional,
        args.reconnect_attempts,
    )
    segment_manifest = _build_segment_manifest(
        segment_id,
        window_manifest,
        monitor_manifest,
        staging_dir,
        cursor,
        args.max_gap_seconds,
    )
    _atomic_write_json(staging_dir / "segment_manifest.json", segment_manifest)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir, final_dir)

    next_cursor = _checkpoint_after(cursor, segment_manifest)
    checkpoint_record = _checkpoint_record(
        window_manifest, next_cursor, segment_manifest
    )
    checkpoint_history_path = (
        state_dir
        / "checkpoints"
        / f"{checkpoint_record['checkpoint_id']}.json"
    )
    _atomic_write_json(checkpoint_history_path, checkpoint_record)
    _atomic_write_json(state_dir / "checkpoint.json", next_cursor)

    if archive is not None:
        archive.upload_segment(final_dir)
        generation_path = state_dir / ".checkpoint.generation"
        expected_generation = (
            int(generation_path.read_text(encoding="utf-8").strip())
            if generation_path.exists()
            else None
        )
        generation = archive.update_checkpoint(next_cursor, expected_generation)
        _atomic_write_text(generation_path, str(generation))
        archive.create_json(
            f"checkpoints/{checkpoint_history_path.name}", checkpoint_record
        )

    result = {
        "status": "segment-complete",
        "window_id": args.window_id,
        "segment_id": segment_id,
        "data_through_utc": next_cursor["data_through"],
        "continuity_status": segment_manifest["continuity_status"],
        "gap_seconds_from_previous": segment_manifest[
            "gap_seconds_from_previous"
        ],
        "segments_completed": next_cursor["segments_completed"],
        "normalized_event_count": next_cursor["normalized_event_count"],
        "research_only": True,
        "orders_allowed": False,
        "promotion_allowed": False,
    }
    print(json.dumps(result, indent=2))
    if args.fail_on_gap and segment_manifest["continuity_status"] in {
        "gap",
        "overlap",
    }:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
