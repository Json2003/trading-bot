"""Base classes and utilities for feature ingestion pipelines."""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class IngestionResult:
    """Container describing the outcome of a pipeline run."""

    name: str
    as_of: dt.datetime
    data_version: str
    row_count: int
    artifact_path: str
    metadata: Dict[str, Any]


class IngestionPipeline(abc.ABC):
    """Abstract base class for feature ingestion pipelines."""

    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abc.abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return a raw dataframe for the pipeline."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply default transformations ensuring mandatory columns exist."""
        if "event_ts" not in df.columns:
            raise ValueError(f"{self.name} pipeline produced data without event_ts column")
        df = df.copy()
        df["as_of"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        df["data_version"] = df.get("data_version", pd.Series([self.version_string()] * len(df)))
        return df

    def version_string(self) -> str:
        return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def run(self) -> pd.DataFrame:
        raw = self.fetch()
        return self.transform(raw)
