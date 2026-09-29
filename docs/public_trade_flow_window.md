# Sustained public trade-flow research window

PR #276 supports a durable, restart-safe collection window for the
normalized Binance USD-M public streams used by the observer.

## Storage contract

The canonical archive is:

`gs://<bucket>/<prefix>/`

Each completed segment is immutable:

`segments/segment-000001/`

It contains the raw normalized observer output, completed-minute summaries,
the original monitor manifest, and a checksummed `segment_manifest.json`.
The controller uploads the segment before advancing `checkpoint.json`. Each segment object is written with
a GCS create-only generation precondition, so retries cannot replace an
already archived raw file.

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

## Starting collection

The hourly workflow is scheduled on the repository's default branch only.
Merge this workflow onto `main` before expecting scheduled runs. Manual
dispatch is restricted to `main` as well.

Before the first run, set the required repository secret and bucket variable
below. Use a new empty GCS prefix for the initial archive. If the selected prefix contains objects but has no trade-flow window manifest, the collector stops before capturing a segment. The workflow uses
`TRADE_FLOW_GCS_PREFIX` when set; otherwise it selects
`research/binance-trade-flow/pr276-btc-eth-90d-2026-09`. Later runs must
reuse the same prefix and window ID. Do not point another window at an
existing prefix. The collector checks its frozen window manifest and refuses
a configuration mismatch.

The workflow validates the secret and bucket before attempting Google
authentication, so missing setup is reported before collection starts. It
does not print the service-account key.

## Required repository configuration

- Secret: `GCP_SERVICE_ACCOUNT_KEY`, with write access to the selected bucket
  and prefix.
- Variable: `TRADE_FLOW_GCS_BUCKET`.

Optional repository variables override defaults: `TRADE_FLOW_GCS_PREFIX`,
`TRADE_FLOW_WINDOW_ID`, `TRADE_FLOW_TARGET_DAYS`, and
`TRADE_FLOW_SEGMENT_SECONDS`. Alternatively, a manual dispatch can supply a
bucket and prefix; it should use the same settings as the scheduled workflow
so it resumes the same window.

The workflow remains research-only. It does not import the broker, access
account credentials, place orders, enable leverage, modify risk settings, or
promote a strategy.
