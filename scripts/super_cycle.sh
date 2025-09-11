#!/usr/bin/env bash
set -euo pipefail

# Nightly learning cycle (paper):
# - Run Superbot with regime promotion gates
# - Optionally rotate model tag by writing to models/active_tag.txt

SYMBOL=${SYMBOL:-"BTC/USDT"}
TIMEFRAME=${TIMEFRAME:-"4h"}
OUTDIR=${OUTDIR:-"artifacts/superbot"}
MODEL_TAG=${MODEL_TAG:-""}

python scripts/superbot.py \
  --symbol "$SYMBOL" --timeframe "$TIMEFRAME" \
  --regime_promotion ${MODEL_TAG:+--set_active_tag "$MODEL_TAG"}

echo "Super cycle complete. Active config at $OUTDIR/active_config.json" >&2

