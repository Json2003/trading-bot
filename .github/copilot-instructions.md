# Copilot Instructions for trading-bot

Purpose: Give AI coding agents the minimum context to be productive in this repo. Keep answers pragmatic and aligned with how this project actually works.

## Architecture at a glance
- Core package: `tradingbot_ibkr/` – modular trading pipeline.
  - `feature_extraction.py` → builds features from OHLCV.
  - `signal_generators.py` → `get_signal_generator(name)` returns functions like `sma_cross`, `enhanced`, `breakout`.
  - `backtest_ccxt.py` → engines and analytics (`aggressive_strategy_backtest`, metrics like Sharpe, drawdown).
  - `models/` → `train_batch.py` (batch ML), `online_trainer.py` (incremental), persisted under `model_store/`.
  - `data/` → `store.py` manages `datafiles/` (bars/trades CSV), `model_store/` paths.
  - Ops helpers: `aggressive_optimize*.py`, `worker.py`, `job_db.py`.
- Server/API: `server.py` (FastAPI) with CORS, rate limiting, mock JWT, WS broadcasting, logs to `server.log`.
- Dashboard: `tradingbot_ibkr/dashboard.py` (Flask) and `dashboard/` clients (electron/flutter minimal scaffolds).
- CLI harness: `scripts/run_backtest.py` to load OHLCV (CSV/CCXT), apply signals, run backtests, and save JSON reports.

## Developer workflows
- Install (Linux/macOS):
  - `./install.sh`
  - Then: `pip install -r requirements.txt -r tradingbot_ibkr/requirements.txt`
- Backtest quickly:
  - CSV: `python scripts/run_backtest.py --source csv --path data/BTCUSDT-1h.csv --strategy sma_cross`
  - Live fetch (CCXT): `python scripts/run_backtest.py --source ccxt --exchange binance --symbol "BTC/USDT" --timeframe 1h --since 2024-01-01`
- Optimization (grid search):
  - `python tradingbot_ibkr/aggressive_optimize.py --symbol ETH/USDT --workers 8 --patience 30`
- Model training (batch):
  - Use `tradingbot_ibkr/models/train_batch.py: train_and_evaluate_models(df, optimize_hyperparams=..., use_optuna=...)`.
  - Note: There is no `scripts/train_lightgbm.py` in this repo; use `models/train_batch.py` instead.
- Server:
  - `python server.py` → REST + WS, mock JWT, state in `STATE` with metrics; logs to `server.log`.
- Tests/validation (no pytest config required):
  - Sanity: `python test_functionality.py`
  - Enhancements: `python test_enhancements.py [--component backtest|optimization|models|data|server]`

## Conventions and gotchas
- Avoid import shadowing: The repo contains files like `pandas.py`/`requests.py` (for demos). In CLIs, import third-party packages via `scripts/run_backtest.py: import_third_party()` or delay imports inside functions. Prefer absolute imports for local modules.
- Data/model layout (managed by `tradingbot_ibkr/data/store.py`):
  - Bars CSV: `tradingbot_ibkr/datafiles/<SYMBOL>_bars.csv`
  - Trade logs: `tradingbot_ibkr/datafiles/trades.csv` and per-symbol `*_trades.csv`
  - Models + jobs: `tradingbot_ibkr/model_store/` (`jobs/`, `jobs_archive/`, `logs/`)
- Model promotion safety: Requires `ALLOW_MODEL_PROMOTE=true` in `.env` AND a file `allow_live_confirm.txt` at repo root (`tradingbot_ibkr/README.md`).
- Backtest signal injection: `scripts/run_backtest.py` joins `signal_generators` outputs onto the OHLCV index before calling `aggressive_strategy_backtest`.
- FastAPI auth tokens in `server.py` are simplified (no real JWT signing). For production, wire `pyjwt` and secure secrets.

## Integration touchpoints
- Exchanges: CCXT (`EXCHANGE`, `PAPER` in environment via `.env` and `load_dotenv`).
- Data ingestion: Binance tools under `tradingbot_ibkr/` and top-level scripts (`binance_vision_to_gcs.py`, `binance_to_gcs.py`, etc.) – Google Cloud Storage integration via `google-cloud-storage`.
- IBKR: `ib_insync`-based live examples (`ibkr_live_*`), requires TWS/Gateway config.
- Dashboard/job runner: Drop JSON jobs in `tradingbot_ibkr/model_store/jobs/`; `worker.py` consumes them and archives to `jobs_archive/` with logs under `logs/`.

## When adding or modifying code
- New CLIs: mirror `scripts/run_backtest.py` structure (argparse, lazy imports, `import_third_party`, JSON summary output path naming).
- Backtests: use vectorized pandas where possible; compute metrics via helpers in `backtest_ccxt.py` to keep consistency with `test_enhancements.py`.
- Data writes: use `tradingbot_ibkr/data/store.py` helpers to ensure consistent file locations and headers.
- Logging: follow module-level `logging.basicConfig(..., FileHandler, StreamHandler)` patterns seen in `server.py`, `backtest_ccxt.py`, `models/train_batch.py`.

## Quick references
- Example: get a signal generator → `from tradingbot_ibkr.signal_generators import get_signal_generator; df = get_signal_generator('sma_cross')(df)`
- Example: batch-train → `from tradingbot_ibkr.models.train_batch import train_and_evaluate_models` and pass a DataFrame with OHLCV index.
- Example: run WS server health check → `curl http://localhost:8000/health` (see `SETUP_GUIDE.md` for more endpoints).

Feedback needed:
- Confirm production JWT requirements and any expected auth flows beyond current stubs.
- Confirm if additional training scripts are desired (e.g., LightGBM); currently, use `models/train_batch.py`.
- Identify any non-default env vars relied upon in deployments (beyond those listed in `SETUP_GUIDE.md`).
