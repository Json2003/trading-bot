#!/usr/bin/env bash
set -euo pipefail

# config
REPO_DIR="$HOME/trading-bot"
CONDA_DIR="$HOME/miniconda"
ENV_NAME="pipeline"
PILOT_SYMBOLS=("BTCUSDT" "ETHUSDT" "ADAUSDT")
SINCE="2021-01"
UNTIL="2021-03"
OUT_ROOT="$HOME/raw_all"
GCS_PREFIX="gs://historical-trade-data/binance/spot/trades"
SA_KEY="$HOME/sa-key.json"   # path to service account JSON you copied earlier; optional if using gcloud auth

# Install prerequisites
sudo apt-get update -y
sudo apt-get install -y git curl build-essential unzip

# Install Miniconda non-interactively if not present
if [ ! -d "${CONDA_DIR}" ]; then
  echo "Installing Miniconda..."
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
fi
export PATH="${CONDA_DIR}/bin:${PATH}"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Create conda env and install binary deps
if ! conda env list | grep -q "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"
conda install -y -c conda-forge numpy pandas pyarrow fastparquet

# Install remaining Python deps from requirements (use --no-deps so conda binary deps are used)
pip install --no-deps -r "${REPO_DIR}/tradingbot_ibkr/requirements.txt"

# If you copied a service account key: set GOOGLE_APPLICATION_CREDENTIALS for gsutil and client libs
if [ -f "${SA_KEY}" ]; then
  export GOOGLE_APPLICATION_CREDENTIALS="${SA_KEY}"
fi

# Make sure gsutil is available (gcloud sdk generally present); otherwise install google-cloud-storage via pip
if ! command -v gsutil >/dev/null 2>&1; then
  pip install google-cloud-storage
fi

mkdir -p "${OUT_ROOT}"

# Pilot: list-only discovery for USDT symbols (sanity)
python "${REPO_DIR}/tradingbot_ibkr/scripts/binance_download_all.py" --since "${SINCE}" --until "${UNTIL}" --symbols-regex ".*USDT$" --list-only --out "${OUT_ROOT}"

# Pilot: download three symbols (serial for clearer timing) with remove-zip
for sym in "${PILOT_SYMBOLS[@]}"; do
  echo "Downloading ${sym} ${SINCE}..${UNTIL} -> ${OUT_ROOT}"
  python "${REPO_DIR}/tradingbot_ibkr/scripts/binance_vision_full_download.py" \
    --symbol "${sym}" --since "${SINCE}" --until "${UNTIL}" --out "${OUT_ROOT}" --threads 4 --remove-zip
done

# Merge+convert BTC to parquet
echo "Merging & converting BTCUSDT to parquet"
python "${REPO_DIR}/tradingbot_ibkr/scripts/binance_vision_full_download.py" \
  --symbol BTCUSDT --out "${OUT_ROOT}" --merge parquet --merge-only --merged-name BTCUSDT_pilot

# Produce manifest: rowcounts and sha256 for the parquet
PARQUET_FILE="${OUT_ROOT}/BTCUSDT_pilot.parquet"
if [ -f "${PARQUET_FILE}" ]; then
  python - <<PY
import pandas as pd, hashlib, json
p = "${PARQUET_FILE}"
df = pd.read_parquet(p)
rows = len(df)
h = hashlib.sha256(open(p,'rb').read()).hexdigest()
print("rows:", rows)
print("sha256:", h)
with open("${OUT_ROOT}/BTCUSDT_pilot_manifest.json","w") as f:
    json.dump({"file":p,"rows":rows,"sha256":h}, f)
PY
else
  echo "Parquet not found: ${PARQUET_FILE}" >&2
fi

# Upload parquet and manifest to GCS
echo "Uploading parquet and manifest to ${GCS_PREFIX}/pilot/"
gsutil -m cp "${PARQUET_FILE}" "${OUT_ROOT}/BTCUSDT_pilot_manifest.json" "${GCS_PREFIX}/pilot/"

echo "Pilot complete. Check ${GCS_PREFIX}/pilot/ for results."
