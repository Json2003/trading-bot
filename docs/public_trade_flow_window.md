# Sustained public trade-flow research window

PR #276 now supports a durable, restart-safe collection window for the
normalized Binance USD-M public streams used by the observer.

## Storage contract

The canonical archive is:

`gs://<bucket>/<prefix>/`

Each completed segment is immutable:

`segments/segment-000001/`

It contains the raw normalized observer output, completed-minute summaries,
the original monitor manifest, and a checksummed `segment_manifest.json`.
The controller uploads the segment before advancing `checkpoint.json`.

The controller also writes immutable files under `checkpoints/`. Each
checkpoint records `data_through_utc`. A future analysis must pin one
checkpoint and must not read newer segments during discovery or tuning; those
newer segments remain untouched confirmation data.

## Restart behavior

A fresh CI runner downloads only the window manifest, checkpoint history, and
segment manifests. Raw prior segments are not copied into the runner. If a
run stops after segment upload but before the checkpoint update, the next run
recovers the finalized segment manifest and advances the cursor in sequence.
An abandoned staging directory is preserved under `recovery/`; it is never
silently counted as observed coverage.

Any minute gap or overlap is recorded in the segment manifest. The sustained
window must not be called continuous unless its checkpoint history has zero
gaps and zero overlaps.

## Required repository configuration

- Secret: `GCP_SERVICE_ACCOUNT_KEY`, with write access to the selected bucket
  and prefix.
- Variable: `TRADE_FLOW_GCS_BUCKET`.
- Optional variables: `TRADE_FLOW_GCS_PREFIX`,
  `TRADE_FLOW_WINDOW_ID`, `TRADE_FLOW_TARGET_DAYS`, and
  `TRADE_FLOW_SEGMENT_SECONDS`.

The workflow remains research-only. It does not import the broker, access
account credentials, place orders, enable leverage, modify risk settings, or
promote a strategy.
