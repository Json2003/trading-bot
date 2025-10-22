#!/usr/bin/env bash
<<<<<<< HEAD
# Simple script to set up a Python virtual environment and install dependencies for the Splitstar Operations Console.
set -e
=======
# Simple script to set up a Python virtual environment and install dependencies for the trading bot.
set -euo pipefail

VENV_DIR="venv"
>>>>>>> origin/main

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
. "${VENV_DIR}/bin/activate"

check_pypi_access() {
    python - <<'PY'
import sys
import urllib.request

URL = "https://pypi.org/simple/pip/"
try:
    with urllib.request.urlopen(URL, timeout=5):
        pass
except Exception as exc:  # pragma: no cover - executed only when network blocked
    print(exc)
    sys.exit(1)
PY
}

if check_pypi_access; then
    # Upgrade pip and install requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r tradingbot_ibkr/requirements.txt
    echo "\nInstallation complete. Activate the environment with 'source ${VENV_DIR}/bin/activate'."
else
    cat <<MSG

Network connectivity to PyPI could not be established. The virtual environment
was created at '${VENV_DIR}', but dependency installation was skipped.
Please install the requirements manually once network access is available.
MSG
fi
