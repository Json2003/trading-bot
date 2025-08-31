#!/usr/bin/env bash
# Upload local datafiles to GCS bucket using gsutil.
# Usage: edit BUCKET and LOCAL_DIR or pass them as env vars.

set -euo pipefail

BUCKET=${BUCKET:-historicaltradedataromantradebot}
LOCAL_DIR=${LOCAL_DIR:-$(pwd)/tradingbot_ibkr/datafiles}
DEST_PATH=${DEST_PATH:-data}

echo "Uploading $LOCAL_DIR to gs://$BUCKET/$DEST_PATH"

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil not found in PATH. Run this in Cloud Shell or install Google Cloud SDK." >&2
  exit 2
fi

gsutil -m rsync -r "$LOCAL_DIR" "gs://$BUCKET/$DEST_PATH"
echo "Upload complete."
