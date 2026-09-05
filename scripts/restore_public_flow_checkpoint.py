#!/usr/bin/env python3
"""Read only the completed-minute archive covered by one explicit GCS checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def restore_checkpoint(bucket, prefix: str, checkpoint_name: str, destination: Path) -> dict:
    if not re.fullmatch(r"checkpoints/[A-Za-z0-9_.-]+\.json", checkpoint_name):
        raise ValueError("Pin an immutable checkpoints/<name>.json, not checkpoint.json")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Archive destination must be empty; do not mix checkpoints")

    def read(relative: str) -> bytes:
        name = f"{prefix.strip('/')}/{relative}" if prefix.strip('/') else relative
        return bucket.blob(name).download_as_bytes()

    checkpoint_bytes = read(checkpoint_name)
    checkpoint = json.loads(checkpoint_bytes)
    if checkpoint.get("continuity_status") != "continuous":
        raise ValueError("Checkpoint contains gaps or overlaps")
    last = checkpoint["inclusive_segment_id"]
    if not re.fullmatch(r"segment-\d{6}", last):
        raise ValueError("Invalid checkpoint segment ID")
    count = int(last.split("-")[1])
    if count < 1 or checkpoint["segments_completed"] != count:
        raise ValueError("Checkpoint segment count mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    # Explicit IDs only: never list the bucket or fetch newer segments.
    for number in range(1, count + 1):
        segment = f"segment-{number:06d}"
        relative = f"segments/{segment}"
        manifest_bytes = read(f"{relative}/segment_manifest.json")
        manifest = json.loads(manifest_bytes)
        if (manifest["segment_id"] != segment
                or manifest["window_id"] != checkpoint["window_id"]):
            raise ValueError("Segment identity mismatch")
        if manifest["continuity_status"] not in {"initial", "continuous"}:
            raise ValueError("Segment contains gaps or overlaps")
        if number == count and manifest["observed_end_exclusive_utc"] != checkpoint["data_through_utc"]:
            raise ValueError("Checkpoint end does not match final segment")
        filename = "completed_minute_flow.csv"
        expected = manifest["files"][filename]
        payload = read(f"{relative}/{filename}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError("Completed-minute archive checksum mismatch")
        target = destination / relative
        target.mkdir(parents=True)
        (target / filename).write_bytes(payload)
        (target / "segment_manifest.json").write_bytes(manifest_bytes)
        hashes[segment] = digest
    return {
        "checkpoint": checkpoint,
        "checkpoint_object": checkpoint_name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "bucket": bucket.name,
        "prefix": prefix,
        "completed_minute_sha256": hashes,
        "research_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="research/binance-trade-flow")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    from google.cloud import storage

    provenance = restore_checkpoint(storage.Client().bucket(args.bucket), args.prefix,
                                    args.checkpoint, args.destination)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
