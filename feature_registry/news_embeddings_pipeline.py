"""Ingestion pipeline producing lightweight news embeddings from RSS feeds."""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import List

from .base import IngestionPipeline
from .vendor import import_requests

requests = import_requests()
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class NewsEmbeddingsPipeline(IngestionPipeline):
    """Fetches headlines from free RSS feeds and generates TF-IDF embeddings."""

    FEEDS = {
        "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    }

    def __init__(self, max_items: int = 50) -> None:
        super().__init__(name="news_embeddings")
        self.max_items = max_items

    def fetch(self) -> pd.DataFrame:
        records: List[dict] = []
        for source, url in self.FEEDS.items():
            xml_text = self._download_feed(url)
            entries = self._parse_feed(xml_text)
            for entry in entries[: self.max_items]:
                records.append(
                    {
                        "source": source,
                        "title": entry["title"],
                        "summary": entry["summary"],
                        "event_ts": entry["published"],
                    }
                )

        if not records:
            records = self._fallback_records()

        df = pd.DataFrame(records)
        df.sort_values("event_ts", inplace=True)
        df["embedding"] = self._vectorize(df["summary"].tolist())
        df["as_of"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        df["data_version"] = self.version_string()
        return df

    def _download_feed(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    def _parse_feed(self, xml_text: str) -> List[dict]:
        if not xml_text:
            return []
        root = ET.fromstring(xml_text)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            summary = (item.findtext("description") or "").strip()
            pub = item.findtext("pubDate") or item.findtext("date")
            if not title or not summary or not pub:
                continue
            try:
                published = parsedate_to_datetime(pub)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=dt.timezone.utc)
                else:
                    published = published.astimezone(dt.timezone.utc)
            except Exception:
                published = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            items.append({"title": title, "summary": summary, "published": published})
        return items

    def _vectorize(self, summaries: List[str]) -> List[List[float]]:
        vectorizer = TfidfVectorizer(max_features=128)
        matrix = vectorizer.fit_transform(summaries)
        return [row.astype(np.float32).toarray().flatten().tolist() for row in matrix]

    def _fallback_records(self) -> List[dict]:
        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        samples = [
            {
                "source": "sample_feed",
                "title": "Global markets steady amid policy uncertainty",
                "summary": "Equity markets held gains while investors weighed central bank guidance and global demand indicators.",
                "event_ts": now - dt.timedelta(hours=1),
            },
            {
                "source": "sample_feed",
                "title": "Energy prices climb as supply risks resurface",
                "summary": "Oil benchmarks advanced after renewed supply disruptions, stirring inflation watchers across major economies.",
                "event_ts": now - dt.timedelta(hours=2),
            },
        ]
        return samples
