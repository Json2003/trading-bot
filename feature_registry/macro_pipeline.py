"""Ingestion pipeline for macroeconomic indicators using free public APIs."""

from __future__ import annotations

import datetime as dt
from typing import List

from .base import IngestionPipeline
from .vendor import import_requests

requests = import_requests()
import pandas as pd


class MacroEconomicPipeline(IngestionPipeline):
    """Fetches macroeconomic indicators from the World Bank API."""

    SERIES = {
        "USA": {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
            "FP.CPI.TOTL.ZG": "cpi_inflation_pct",
        },
        "GBR": {
            "SL.UEM.TOTL.ZS": "unemployment_rate_pct",
        },
    }

    def __init__(self, start_year: int = 2015, end_year: int | None = None) -> None:
        super().__init__(name="macro_worldbank")
        self.start_year = start_year
        self.end_year = end_year or dt.datetime.utcnow().year

    def fetch(self) -> pd.DataFrame:
        records: List[dict] = []
        for country, series_map in self.SERIES.items():
            for series_id, alias in series_map.items():
                data = self._fetch_series(country, series_id)
                for entry in data:
                    year = entry.get("date")
                    value = entry.get("value")
                    if year is None or value is None:
                        continue
                    year = int(year)
                    if year < self.start_year or year > self.end_year:
                        continue
                    event_ts = dt.datetime(year, 12, 31, tzinfo=dt.timezone.utc)
                    records.append(
                        {
                            "country": country,
                            "series_id": series_id,
                            "feature": alias,
                            "value": float(value),
                            "event_ts": event_ts,
                        }
                    )

        if not records:
            records = self._fallback_records()

        df = pd.DataFrame(records)
        df.sort_values("event_ts", inplace=True)
        df["as_of"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        df["data_version"] = self.version_string()
        return df

    def _fetch_series(self, country: str, series_id: str) -> list[dict]:
        url = (
            f"https://api.worldbank.org/v2/country/{country}/indicator/{series_id}"
            f"?format=json&per_page=1000"
        )
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        # payload format: [metadata, data]
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        data = payload[1] or []
        if not isinstance(data, list):
            return []
        return data

    def _fallback_records(self) -> list[dict]:
        now = dt.datetime.utcnow().year
        event_ts = dt.datetime(now, 12, 31, tzinfo=dt.timezone.utc)
        return [
            {
                "country": "USA",
                "series_id": "NY.GDP.MKTP.KD.ZG",
                "feature": "gdp_growth_pct",
                "value": 2.1,
                "event_ts": event_ts,
            },
            {
                "country": "USA",
                "series_id": "FP.CPI.TOTL.ZG",
                "feature": "cpi_inflation_pct",
                "value": 3.2,
                "event_ts": event_ts,
            },
        ]
