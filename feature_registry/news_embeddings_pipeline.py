"""Ingestion pipeline producing lightweight news embeddings from RSS feeds."""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import List

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import IngestionPipeline


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
            raise RuntimeError("NewsEmbeddingsPipeline fetched no articles")

        df = pd.DataFrame(records)
        df.sort_values("event_ts", inplace=True)
        df["embedding"] = self._vectorize(df["summary"].tolist())
        df["as_of"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        df["data_version"] = self.version_string()
        return df

    def _download_feed(self, url: str) -> str:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text

    def _parse_feed(self, xml_text: str) -> List[dict]:
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
