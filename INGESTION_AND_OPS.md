# Ingestion & Ops — Splitstar Operations Console

This README is a concise operations checklist and mapping for the Splitstar
Operations Console data ingestion pipelines. It covers bootstrap commands to
run on a VM, local re-activation steps, a map of ingestion scripts (inputs →
outputs), common troubleshooting, and alternative exchanges for CCXT.

## VM bootstrap checklist (minimal)

Run these steps on a VM (Ubuntu 22.04/24.04 recommended) to create a reproducible ingestion node.

1. Create a non-root user and update the system

   sudo apt-get update && sudo apt-get upgrade -y

2. Install runtime tools

   sudo apt-get install -y git curl build-essential unzip python3 python3-venv python3-pip

3. Clone repo and create Python environment

   git clone <REPO_URL> splitstar-operations-console
   cd splitstar-operations-console
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

4. Install system deps (optional via conda)

   Optional: if you prefer conda for binary wheel performance

   Install miniconda and create env then install pyarrow/fastparquet there

5. Install Python dependencies

   pip install -r requirements.txt
   pip install -r tradingbot_ibkr/requirements.txt

6. (If uploading to GCS) Configure GCP credentials

   Use either gcloud auth or a service account key

   gcloud auth login

   OR

   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json

7. Ensure gsutil or google-cloud-storage is available

   If gsutil is available via gcloud, fine. Otherwise

   pip install google-cloud-storage

8. (Optional) Install `ccxt` for live fetches

   pip install ccxt

9. Run the pilot pipeline (quick end-to-end test)

   Run the pipeline startup script which downloads small pilot symbols and uploads to GCS

   bash tradingbot_ibkr/scripts/pipeline_startup.sh

   Or run steps individually (safer, inspect each step)

   python tradingbot_ibkr/scripts/binance_download_all.py --since 2021-01 --until 2021-03 --symbols-regex ".*USDT$" --list-only --out /tmp/raw_all
   python tradingbot_ibkr/scripts/binance_vision_full_download.py --symbol BTCUSDT --since 2021-01 --until 2021-03 --out /tmp/raw_all --threads 4
   python scripts/binance_raw_to_parquet.py --raw /tmp/raw_all --out data/parquet/ohlcv_1m --symbols BTCUSDT

## Minimal local re-activation commands

- Backtest (CSV):

  python scripts/run_backtest.py --source csv --path tradingbot_ibkr/datafiles/BTC_USDT_bars.csv --strategy sma_cross

- Backtest (CCXT live fetch):

  pip install ccxt
  python scripts/run_backtest.py --source ccxt --exchange binance --symbol "BTC/USDT" --timeframe 1h --since 2023-01-01 --until 2023-01-31 --strategy sma_cross

- Train model (LightGBM):

  python tradingbot_ibkr/models/train_batch.py --source csv --path tradingbot_ibkr/datafiles/BTC_USDT_bars.csv --out tradingbot_ibkr/model_store

- Convert raw CSV.GZ -> Parquet

  BINANCE_RAW_DIR=/path/to/raw BINANCE_RAW_DIR_PARQUET_OUT=data/parquet/test python scripts/binance_raw_to_parquet.py --raw /path/to/raw --out data/parquet/test --symbols BTCUSDT

## Ingestion map (script -> inputs -> outputs)

- `tradingbot_ibkr/scripts/binance_vision_full_download.py`
  - Inputs: remote archives at <https://data.binance.vision>
  - Outputs: local raw CSVs or merged CSV/Parquet in `<out>`
  - Activate: `python tradingbot_ibkr/scripts/binance_vision_full_download.py --symbol BTCUSDT --since 2021-01 --until 2021-03 --out /tmp/raw_all`

- `tradingbot_ibkr/scripts/binance_download_all.py`
  - Inputs: ccxt symbol discovery or manual symbol list
  - Outputs: orchestrates per-symbol downloads (calls binance_vision_full_download)

- `tradingbot_ibkr/binance_vision_to_gcs.py`
  - Inputs: local raw or merged files
  - Outputs: uploaded to GCS bucket
  - Activate: `python tradingbot_ibkr/binance_vision_to_gcs.py --bucket my-bucket --symbol BTCUSDT --out-path raw/binance`

- `tradingbot_ibkr/binance_trade_dump_ingest.py`
  - Inputs: Binance trade dump files
  - Outputs: normalized trade CSVs and optional OHLCV

- `scripts/binance_raw_to_parquet.py`
  - Inputs: BINANCE_RAW_DIR (local *.csv.gz)
  - Outputs: Partitioned Parquet under PARQUET_OUT (default `data/parquet/ohlcv_1m`)
  - Activate: `BINANCE_RAW_DIR=data/raw/binance/spot PARQUET_OUT=data/parquet/test python scripts/binance_raw_to_parquet.py --symbols BTCUSDT`

- `scripts/run_backtest.py`
  - Inputs: `--source csv` reads `tradingbot_ibkr/datafiles/<SYMBOL>_bars.csv`; `--source ccxt` fetches via CCXT
  - Outputs: backtest JSON reports in repo root (backtest_*.json)

- `tradingbot_ibkr/models/train_batch.py`
  - Inputs: OHLCV via CSV or CCXT
  - Outputs: model artifacts and reports under `tradingbot_ibkr/model_store/`

## Troubleshooting & notes

- CCXT & exchange access:
  - If `ModuleNotFoundError: No module named 'ccxt'` -> `pip install ccxt`
  - If Binance returns HTTP 451 (restricted location), run on a VM in a supported region or use a different exchange.
- GCS uploads:
  - `gsutil` requires `gcloud` or `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON.
- Large downloads:
  - The `pipeline_startup.sh` is intended for a dedicated VM. It installs conda and downloads potentially many TBs if misused—run for small date ranges first.

## Alternative exchanges for CCXT live fetch

If Binance is geo-blocked in your environment, switch to an alternative exchange supported by CCXT that provides OHLCV, e.g.:

- `bybit` (spot & derivatives) — id `bybit`
- `kucoin` — id `kucoin`
- `kraken` — id `kraken`
- `coinbasepro` / `coinbase` — id `coinbasepro`/`coinbase`
- `okx` — id `okx`

Usage example (replace `--exchange binance`):

  python scripts/run_backtest.py --source ccxt --exchange bybit --symbol "BTC/USDT" --timeframe 1h --since 2024-08-30 --until 2024-09-01 --strategy sma_cross

Notes on exchange choice:

- Check exchange rate limits and API key requirements for private endpoints (OHLCV is public on most exchanges).
- Some exchanges use different symbol formats (e.g., `BTC/USDT` vs `BTC/USDT:USDT`); use `tradingbot_ibkr/scripts/binance_download_all.py` style symbol normalization or query `ccxt` for available markets.

## Ops checklist for safe promotion to live

- Ensure `ALLOW_MODEL_PROMOTE=true` in environment and create `allow_live_confirm.txt` in repo root to enable model promotions.
- Run backtests and review backtest_*.json outputs before applying model to live.
- Use `tradingbot_ibkr/worker.py` to manage jobs and `tradingbot_ibkr/model_store/jobs/` to queue promotion jobs.

---

If you want, I can also:

- Create a runnable `bootstrap_vm.sh` that automates the VM bootstrap steps above (I can add a conservative, commented script).
- Produce a compact PDF exported from this Markdown.
- Run a small CCXT backtest against an alternative exchange here (I can install any exchange-specific deps if needed).

Which follow-up should I do now?
