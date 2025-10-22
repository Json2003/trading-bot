"""Service orchestrating feature ingestion pipelines."""

from __future__ import annotations

import datetime as dt
from typing import Iterable, List, Sequence

import pandas as pd

from .base import IngestionPipeline, IngestionResult
from .storage import FeatureStore


class FeatureRegistryService:
    """Orchestrates pipeline execution and persistence."""

    def __init__(self, pipelines: Sequence[IngestionPipeline], store: FeatureStore | None = None) -> None:
        self.pipelines = list(pipelines)
        self.store = store or FeatureStore()

    def run_all(self, limit: int | None = None) -> List[IngestionResult]:
        results: List[IngestionResult] = []
        for pipeline in self._iter_pipelines(limit):
            df = pipeline.run()
            # ensure required columns
            df = self._enforce_schema(df)
            metadata = self.store.write_snapshot(pipeline.name, df)
            results.append(
                IngestionResult(
                    name=pipeline.name,
                    as_of=dt.datetime.fromisoformat(metadata.as_of),
                    data_version=metadata.data_version,
                    row_count=metadata.row_count,
                    artifact_path=metadata.artifact_path,
                    metadata={
                        "event_start": metadata.event_start,
                        "event_end": metadata.event_end,
                    },
                )
            )
        return results

    def _iter_pipelines(self, limit: int | None) -> Iterable[IngestionPipeline]:
        count = 0
        for pipeline in self.pipelines:
            yield pipeline
            count += 1
            if limit is not None and count >= limit:
                break

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if "as_of" not in df.columns:
            df["as_of"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        if "data_version" not in df.columns:
            df["data_version"] = self._version_string()
        if "event_ts" not in df.columns:
            raise ValueError("event_ts column required for feature store compatibility")
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
        df["as_of"] = pd.to_datetime(df["as_of"], utc=True, errors="coerce")
        return df

    def _version_string(self) -> str:
        return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
