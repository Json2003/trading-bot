"""Storage utilities for persisting feature dataframes and metadata."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


DEFAULT_STORE_PATH = Path("data") / "feature_store"
DEFAULT_CACHE_PATH = DEFAULT_STORE_PATH / "cache"


@dataclass
class SnapshotMetadata:
    name: str
    as_of: str
    data_version: str
    event_start: str
    event_end: str
    row_count: int
    artifact_path: str


class FeatureStore:
    """Simple feature store that writes parquet snapshots and maintains metadata."""

    def __init__(self, root: Path = DEFAULT_STORE_PATH) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        DEFAULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, pipeline_name: str, data_version: str) -> Path:
        return self.root / pipeline_name / f"{data_version}.parquet"

    def _metadata_path(self, pipeline_name: str) -> Path:
        return self.root / pipeline_name / "metadata.json"

    def write_snapshot(self, pipeline_name: str, df: pd.DataFrame) -> SnapshotMetadata:
        if "data_version" not in df.columns:
            raise ValueError("data_version column missing from dataframe")
        if "event_ts" not in df.columns:
            raise ValueError("event_ts column missing from dataframe")
        data_version = str(df["data_version"].iloc[0])
        path = self._artifact_path(pipeline_name, data_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        metadata = SnapshotMetadata(
            name=pipeline_name,
            as_of=str(df["as_of"].max()),
            data_version=data_version,
            event_start=str(df["event_ts"].min()),
            event_end=str(df["event_ts"].max()),
            row_count=int(len(df)),
            artifact_path=str(path),
        )

        self._append_metadata(metadata)
        self._update_cache(pipeline_name, df)
        return metadata

    def _append_metadata(self, metadata: SnapshotMetadata) -> None:
        meta_path = self._metadata_path(metadata.name)
        records: Iterable[Dict[str, str]]
        if meta_path.exists():
            records = json.loads(meta_path.read_text())
        else:
            records = []
        updated = list(records) + [asdict(metadata)]
        meta_path.write_text(json.dumps(updated, indent=2))

    def _update_cache(self, pipeline_name: str, df: pd.DataFrame) -> None:
        """Persist the latest snapshot to a lightweight cache (JSON)."""
        cache_path = DEFAULT_CACHE_PATH / f"{pipeline_name}.json"
        sample = df.sort_values("event_ts").tail(100)
        cache_payload = {
            "pipeline": pipeline_name,
            "updated_at": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": sample.to_dict(orient="records"),
        }
        cache_path.write_text(json.dumps(cache_payload, indent=2))
