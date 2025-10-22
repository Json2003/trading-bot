# Feature Registry

This module introduces a lightweight feature-store workflow that can be expanded
into a broader “data breadth” platform. It provides two ingestion pipelines that
pull free public data sources, normalise them with strict `event_ts` semantics,
and persist versioned Parquet snapshots alongside a small JSON cache for
low-latency lookups.

## Components

- `feature_registry/macro_pipeline.py` — fetches macroeconomic indicators from
  the World Bank API (GDP growth, CPI inflation, unemployment). When the public
  API is unavailable it falls back to a synthetic sample so downstream systems
  retain predictable schemas.
- `feature_registry/news_embeddings_pipeline.py` — reads RSS feeds (BBC World
  and Reuters Business) and generates compact TF‑IDF embeddings. Feed outages
  degrade gracefully to a bundled sample.
- `feature_registry/storage.py` — writes versioned Parquet snapshots under
  `data/feature_store/<pipeline>/<data_version>.parquet` and materialises a JSON
  cache with the most recent rows to emulate a Redis/BQ low-latency store.
- `feature_registry/registry_service.py` — orchestrates pipeline execution,
  enforces schema requirements (`event_ts`, `as_of`, `data_version`), and tracks
  metadata (row counts, event horizons).
- `feature_registry/run_demo_ingest.py` — CLI entry point that wires the macro
  and news pipelines together for an end-to-end demo run.

## Usage

```bash
source .venv/bin/activate
python -m feature_registry.run_demo_ingest --max-news 5 --start-year 2018
```

Successful execution prints a JSON summary and materialises outputs under
`data/feature_store/`:

- `macro_worldbank/<data_version>.parquet`
- `news_embeddings/<data_version>.parquet`
- `cache/macro_worldbank.json`
- `cache/news_embeddings.json`

Each snapshot contains `event_ts`, `as_of`, `data_version`, and domain-specific
features, ensuring downstream consumers can enforce no-lookahead joins.

## Extending

- Register additional pipelines by subclassing `IngestionPipeline` and passing
  them into `FeatureRegistryService`.
- Swap the JSON cache with Redis/BigQuery clients while keeping the same
  contract (`pipeline`, `updated_at`, `rows`).
- Use the stored metadata (event start/end, version id) to power feature
  freshness dashboards or automated backfill triggers.
