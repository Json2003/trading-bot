
#!/usr/bin/env bash
set -euo pipefail

# Launch the FastAPI trading bot server locally.
# - Activates local virtualenv if found
# - Starts uvicorn in reload mode on PORT (default 8000)
# - Writes PID to run.pid and logs to server.out

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
PORT="${PORT:-8000}"

echo "Starting Trading Bot server on port ${PORT}..."
python -m uvicorn server:app --host 0.0.0.0 --port "${PORT}" --reload \
  > server.out 2>&1 &
PID=$!
echo "${PID}" > run.pid
echo "Server PID ${PID}. Logs: ${REPO_ROOT}/server.out"

URL="http://127.0.0.1:${PORT}/health"
echo "Probing ${URL}..."
# Try a quick probe (best-effort)
if command -v curl >/dev/null 2>&1; then
  sleep 1
  curl -fsS "${URL}" || true
fi

# Try to open a browser if available (best-effort)
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://127.0.0.1:${PORT}/docs" || true
elif [[ -n "${BROWSER:-}" ]]; then
  "${BROWSER}" "http://127.0.0.1:${PORT}/docs" || true
fi

echo "Done. Visit http://127.0.0.1:${PORT}/docs"
