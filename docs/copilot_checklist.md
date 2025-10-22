# Copilot Instructions — Printable Checklist

Goal: Make AI agents productive fast in THIS repo. Keep actions concise and practical.

## 0) Setup

- [ ] Install: `./install.sh` then `pip install -r requirements.txt -r tradingbot_ibkr/requirements.txt`
- [ ] Create `.env` (or export): `PAPER`, `EXCHANGE`, `BACKTEST_INTERVAL`, `CONTINUOUS_BACKTEST`, optional CCXT keys
- [ ] Use lazy third‑party imports in CLIs (avoid shadowing from files like `pandas.py`/`requests.py`)

## 1) Architecture quick map

- [ ] Core (`tradingbot_ibkr/`)
  - [ ] Features: `feature_extraction.py`
  - [ ] Signals: `signal_generators.py` → `get_signal_generator('sma_cross'|'enhanced'|'breakout')`
  - [ ] Backtests/metrics: `backtest_ccxt.py` (`aggressive_strategy_backtest`, Sharpe, drawdown)
  - [ ] Models: `models/train_batch.py`, `models/online_trainer.py` → artifacts in `model_store/`
  - [ ] Data paths: `data/store.py`
  - [ ] Ops: `aggressive_optimize*.py`, `worker.py`, `job_db.py`
- [ ] Server/API: `server.py` (FastAPI; CORS; rate limit; mock JWT; WS broadcast; logs → `server.log`)
- [ ] Dashboard stubs: `tradingbot_ibkr/dashboard.py`, `dashboard/`
- [ ] CLI: `scripts/run_backtest.py` (CSV/CCXT → signals → backtest → JSON report)

## 2) Back‑end — APIs & jobs

- [ ] Health: `GET /health` (server.py)
- [ ] Jobs: drop JSON into `tradingbot_ibkr/model_store/jobs/` (consumed by `worker.py`, archived to `jobs_archive/`)
- [ ] Live loop (when `PAPER=false`): background eval per `BACKTEST_INTERVAL` (add if missing)
- [ ] Settings (add if missing): `GET/POST /settings` for toggles (PAPER, STRATEGY, RISK, EXCHANGE)
- [ ] Status stream: WS broadcast in `server.py` for metrics/feed status

## 3) Front‑end — Management Workspace (/manage)

 Build a single page (template or SPA) that:

- [ ] Toggles (bound to `/settings`): PAPER, CONTINUOUS_BACKTEST, STRATEGY, RISK, EXCHANGE
- [ ] Status cards: feeds (latency/last bar/error rate), accounts, trades (PnL/win rate), models (id/last train/promotion gate)
- [ ] Charts: equity curve + drawdown; live OHLCV with signals
- [ ] Actions: Run backtest (N bars), Train batch, Promote model (requires env + confirm file)
- [ ] UX: modern responsive grid; banner for current mode (Paper vs Live)

## 4) Parameters & safety

- [ ] `PAPER=true|false`, `CONTINUOUS_BACKTEST=true|false`, `BACKTEST_INTERVAL=60`
- [ ] `STRATEGY=sma_cross|enhanced|breakout` (via `get_signal_generator`)
- [ ] Risk knobs (if used): `RISK_PCT`, `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`
- [ ] Promotion guard: `ALLOW_MODEL_PROMOTE=true` AND file `allow_live_confirm.txt` at repo root

## 5) Data ingestion & feeds

- [ ] Local CSVs: `tradingbot_ibkr/datafiles/*_bars.csv` (quick backtests)
- [ ] CCXT fetch: `scripts/run_backtest.py --source ccxt --exchange <ex> --symbol "<BASE/QUOTE>" --timeframe 1h --since <ISO>`
- [ ] Binance Vision → Parquet: `scripts/binance_raw_to_parquet.py` (can aggregate trades → OHLCV)
- [ ] GCS integration: `google-cloud-storage` (bucket envs required)

## 6) Common commands

- [ ] Backtest CSV: `python scripts/run_backtest.py --source csv --path tradingbot_ibkr/datafiles/BTC_USDT_bars.csv --strategy sma_cross`
- [ ] Backtest CCXT: `python scripts/run_backtest.py --source ccxt --exchange binance --symbol "BTC/USDT" --timeframe 1h --since 2024-01-01`
- [ ] Optimize: `python tradingbot_ibkr/aggressive_optimize.py --symbol ETH/USDT --workers 8 --patience 30`
- [ ] Train batch: `from tradingbot_ibkr.models.train_batch import train_and_evaluate_models`
- [ ] Start server: `python server.py` → `curl http://localhost:8000/health`

## 7) Conventions & gotchas

- [ ] No `scripts/train_lightgbm.py` (use `models/train_batch.py`)
- [ ] Use `data/store.py` for read/write paths; avoid hardcoded local paths
- [ ] Metrics: use helpers in `backtest_ccxt.py` to match tests
- [ ] Logging: mirror `server.py`/`backtest_ccxt.py` (file + stream)
- [ ] Avoid import shadowing; prefer lazy third‑party imports in CLIs

## 8) PR checklist

- [ ] Repro steps (install/run/test) included; JSON reports saved
- [ ] Toggles implemented via `/settings` and visible in UI
- [ ] Metrics align with `backtest_ccxt.py`; no secrets committed
- [ ] Outputs under `model_store/` and `datafiles/`; paths via `data/store.py`

## 9) Roadmap alignment — Splitstar Development

- [ ] External plan location (Windows path): `C:\\Users\\j-mga\\OneDrive\\Documents\\GitHub\\Wcoin`
- [ ] Read synced milestones: `docs/splitstar_plan.md`
- [ ] Reference Splitstar milestone IDs in PR titles/descriptions
- [ ] Priorities: `/manage` workspace, continuous loop, ingestion health, promotion safety
