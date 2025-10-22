#!/usr/bin/env python3
"""Run macro and news ingestion pipelines and persist to the feature store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .macro_pipeline import MacroEconomicPipeline
from .news_embeddings_pipeline import NewsEmbeddingsPipeline
from .registry_service import FeatureRegistryService
from .storage import FeatureStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demo feature ingestions.")
    parser.add_argument("--output-dir", default="data/feature_store", help="Destination for parquet snapshots")
    parser.add_argument("--max-news", type=int, default=30, help="Maximum news items per feed")
    parser.add_argument("--start-year", type=int, default=2015, help="Earliest macro year to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FeatureStore(root=Path(args.output_dir))
    pipelines = [
        MacroEconomicPipeline(start_year=args.start_year),
        NewsEmbeddingsPipeline(max_items=args.max_news),
    ]
    service = FeatureRegistryService(pipelines, store=store)
    results = service.run_all()
    payload = []
    for result in results:
        entry = dict(result.__dict__)
        entry["as_of"] = result.as_of.isoformat()
        payload.append(entry)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
