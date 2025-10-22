#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/superbot

# Learn from recent market using a regime hint
python scripts/superbot.py \
  --symbol "BTC/USDT" --timeframe 4h \
  --since 2023-10-01 --until 2024-03-01 \
  --oos_since 2022-05-01 --oos_until 2022-12-31 \
  --fees_bps 10 --slip_bps 5 --risk_per_trade 0.005 \
  --allow_short --regime_hint bull --outdir artifacts/superbot

echo "Super cycle completed; active_config.json updated."
