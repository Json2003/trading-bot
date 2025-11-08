#!/usr/bin/env bash
set -euo pipefail

# bootstrap_vm.sh — conservative VM bootstrap for Splitstar Operations Console ingestion pipeline
# Usage:
#   ./bootstrap_vm.sh --dry-run       # prints steps but doesn't run destructive installs
#   sudo ./bootstrap_vm.sh --apply    # runs steps (requires sudo for apt installs)

DRY_RUN=true
APPLY=false
REPO_URL="$(git remote get-url origin 2>/dev/null || echo '<REPO_URL>')"
OUT_ROOT="$HOME/raw_all"
ENV_NAME="pipeline"
CONDA_DIR="$HOME/miniconda"

function usage(){
  cat <<EOF
Usage: $0 [--dry-run|--apply] [--out /path/to/out]

Options:
  --dry-run    (default) show steps without making system changes
  --apply      perform actions (will use sudo for apt installs and system-level steps)
  --out PATH   change default OUT_ROOT (default: $OUT_ROOT)
  --help       print this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift;;
    --apply) DRY_RUN=false; APPLY=true; shift;;
    --out) OUT_ROOT="$2"; shift 2;;
    --help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "Bootstrap script — DRY_RUN=$DRY_RUN, APPLY=$APPLY"

echo "Repository URL: $REPO_URL"

echo "1) System update & prerequisites"
CMDS=(
  "sudo apt-get update -y"
  "sudo apt-get install -y git curl build-essential unzip python3 python3-venv python3-pip"
)
for c in "${CMDS[@]}"; do
  if [ "$DRY_RUN" = true ]; then
    echo "  [DRY] $c"
  else
    echo "  RUN: $c"
    eval "$c"
  fi
done

# Clone repo if not present
if [ ! -d "$PWD/.git" ]; then
  if [ "$DRY_RUN" = true ]; then
    echo "  [DRY] git clone $REPO_URL splitstar-operations-console && cd splitstar-operations-console"
  else
    git clone "$REPO_URL" splitstar-operations-console
    cd splitstar-operations-console
  fi
else
  echo "  repo appears to be present in $PWD"
fi

# Create Python venv
if [ "$DRY_RUN" = true ]; then
  echo "  [DRY] python3 -m venv .venv"
  echo "  [DRY] source .venv/bin/activate"
  echo "  [DRY] python -m pip install --upgrade pip"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
fi

# Conditionally offer conda setup (optional path)
cat <<EOF

Note: This script will NOT auto-install Miniconda by default. If you need conda for binary packages
(run the heavy pipeline or on machines without prebuilt wheels), rerun after installing Miniconda.

EOF

# Install Python requirements (conservative: install core-only first)
REQ_CMD="pip install --upgrade pip && pip install -r requirements.txt -r tradingbot_ibkr/requirements.txt"
if [ "$DRY_RUN" = true ]; then
  echo "  [DRY] $REQ_CMD"
else
  echo "  RUN: $REQ_CMD"
  python -m pip install --upgrade pip
  pip install -r requirements.txt -r tradingbot_ibkr/requirements.txt
fi

# GCS prerequisites
cat <<EOF

GCS / Google Cloud Storage:
- If you plan to upload to GCS, either run 'gcloud auth login' on this VM or create a service account
  and set GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json before running pipeline scripts.
- The 'pipeline_startup.sh' script in repo will attempt to use gsutil; this script is conservative and
  will not call gsutil unless you explicitly do so.

EOF

# Create out root dir
if [ "$DRY_RUN" = true ]; then
  echo "  [DRY] mkdir -p $OUT_ROOT"
else
  mkdir -p "$OUT_ROOT"
fi

# Final instructions
cat <<EOF

Bootstrap dry-run completed. Next recommended manual steps:
 - Inspect the repo and the 'tradingbot_ibkr/scripts/pipeline_startup.sh' script.
 - To perform pilot downloads manually:
     python tradingbot_ibkr/scripts/binance_download_all.py --since 2021-01 --until 2021-03 --symbols-regex ".*USDT$" --list-only --out $OUT_ROOT
     python tradingbot_ibkr/scripts/binance_vision_full_download.py --symbol BTCUSDT --since 2021-01 --until 2021-03 --out $OUT_ROOT --threads 4
 - Convert to parquet (local):
     python scripts/binance_raw_to_parquet.py --raw $OUT_ROOT --out data/parquet/ohlcv_1m --symbols BTCUSDT
 - Upload to GCS (if desired):
     gsutil -m cp <files> gs://your-bucket/path/

To actually apply system changes, re-run this script with --apply (run as sudo when apt is used):
  sudo ./bootstrap_vm.sh --apply --out /desired/out

EOF

# Mark complete
if [ "$DRY_RUN" = true ]; then
  echo "Exiting (dry-run)."
else
  echo "Bootstrap APPLY completed (check output above)."
fi
