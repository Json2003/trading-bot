#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source scripts/.env
mkdir -p "${ARTIFACTS_DIR:-artifacts}"

# sanity: ensure keys present
if [[ -z "${KUCOIN_KEY:-}" || -z "${KUCOIN_SECRET:-}" ]]; then
  echo "Live requires KUCOIN_KEY/SECRET in scripts/.env"; exit 1
fi

# kill existing
pkill -f start_crypto.py || true
pkill -f start_equity.py || true
pkill -f risk_manager.py || true

export ENV=live

# start processes
python3 scripts/start_crypto.py  > artifacts/crypto_live.out  2>&1 & echo $! > artifacts/crypto.pid
python3 scripts/start_equity.py  > artifacts/equity_live.out  2>&1 & echo $! > artifacts/equity.pid
python3 scripts/risk_manager.py  > artifacts/risk_live.out    2>&1 & echo $! > artifacts/risk.pid
python3 scripts/health.py        > artifacts/health_live.out  2>&1 & echo $! > artifacts/health.pid

echo "Started LIVE bots (tiny). Monitor artifacts/*.out"

# readiness probe: wait until health returns ok=true (up to 120s)
PORT="${PORT:-8080}"
echo "Waiting for health ok on http://127.0.0.1:${PORT} ..."
ok=0
for i in $(seq 1 60); do
  resp="$(curl -sf "http://127.0.0.1:${PORT}" || true)"
  if echo "$resp" | grep -q '"ok": true'; then
    echo "Health OK: $resp"
    ok=1
    break
  fi
  sleep 2
done
if [ "$ok" -ne 1 ]; then
  echo "Health not OK after timeout. Last response: ${resp:-<none>}"
fi

echo "Tailing logs (Ctrl-C to stop) ..."
tail -n +1 -F artifacts/*_live.out
