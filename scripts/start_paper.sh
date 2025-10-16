#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source scripts/.env || true
mkdir -p "${ARTIFACTS_DIR:-artifacts}"

# kill existing
pkill -f start_crypto.py || true
pkill -f start_equity.py || true
pkill -f risk_manager.py || true

# paper mode
export ENV=paper

# start processes
python3 scripts/start_crypto.py  > artifacts/crypto.out  2>&1 & echo $! > artifacts/crypto.pid
python3 scripts/start_equity.py  > artifacts/equity.out  2>&1 & echo $! > artifacts/equity.pid
python3 scripts/risk_manager.py  > artifacts/risk.out    2>&1 & echo $! > artifacts/risk.pid
python3 scripts/health.py        > artifacts/health.out  2>&1 & echo $! > artifacts/health.pid

echo "Started paper bots. PIDs:"
cat artifacts/*.pid

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
tail -n +1 -F artifacts/*.out
