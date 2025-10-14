# trading bot
tradingbot

## Desktop installation

Run one of the provided scripts to set up a virtual environment and install dependencies.

On Linux or macOS:

```bash
./install.sh
```

On Windows (cmd or PowerShell):

```bat
install.bat
```

This creates a `venv` directory and installs packages from `tradingbot_ibkr/requirements.txt`.

## Working inside GitHub Actions workspaces

When running automation steps in GitHub Actions, the repository is checked out
to the directory pointed at by the `$GITHUB_WORKSPACE` environment variable.
Many reusable workflows begin by ensuring all subsequent commands operate from
that directory:

```bash
pushd "$GITHUB_WORKSPACE" >/dev/null
# ... git commands ...
popd >/dev/null
```

Using `pushd`/`popd` keeps the automation logic self-contained while restoring
the previous directory after any Git operations complete.

## Asset classes

The bot supports multiple asset classes including forex, options, futures,
crypto, and stocks via a unified `AssetClass` enum. Trading scripts and the
engine can adjust risk settings based on the selected class.

## Binance to GCS ingestion

Fetch minute and five-minute klines for BTCUSDT and ETHUSDT across spot and
USDT-margined futures markets and upload them to a Google Cloud Storage bucket:

```bash
python tradingbot_ibkr/binance_to_gcs.py --bucket <bucket-name> \
  --symbols BTCUSDT,ETHUSDT --intervals 1m,5m --markets spot,um \
  --start 2024-01-01T00:00:00 --end 2024-01-01T01:00:00
```

Replace the bucket name and time range as needed.

## Local research data layout

The repository expects large market data and derived features to live outside of
version control. Organize any downloaded datasets under the `data/` directory
using the following convention:

```
data/
  raw/binance/
    spot/BTCUSDT/2022/*.csv.gz
    spot/ETHUSDT/2022/*.csv.gz
  parquet/ohlcv_1m/symbol=BTCUSDT/date=2022-01-01/*.parquet
  parquet/features_1m/symbol=BTCUSDT/date=2022-01-01/*.parquet
```

- `data/raw/` holds the original exchange files (e.g., minute-level Binance
  klines compressed as CSVs).
- `data/parquet/ohlcv_1m/` stores cleaned one-minute OHLCV bars that are ready
  for model training and backtesting.
- `data/parquet/features_1m/` contains engineered features aligned with the
  OHLCV data.

Model checkpoints or other bulky artifacts should be placed beneath the
`artifacts/` directory, which is also ignored by Git:

```
artifacts/
  models/
```

Only keep lightweight samples in the repository itself; all high-volume data
remains in these local folders to keep clones fast and commits reviewable.

## Daily market data fetcher

To generate a lightweight daily dataset for both equities and crypto assets,
use the `scripts/fetch_daily_market_data.py` helper.  The script reads
`configs/daily_data.yaml` to determine the timezone, asset universe, lookback
window, and output location.  By default it downloads the assets listed in the
config (AAPL, MSFT, NVDA, SPY, QQQ along with Bitcoin, Ethereum, Solana,
Binance Coin, and Ripple) and writes timezone-aware parquet files beneath
`data/daily/`.

```
python scripts/fetch_daily_market_data.py
```

Override the lookback window or destination directory at runtime if needed:

```
python scripts/fetch_daily_market_data.py --days 365 --out custom_dir/daily
```

The command prints a short manifest with the saved files for quick inspection
and leaves the manifest in the chosen output directory.

## Sample backtest workflow

For a quick smoke-test of the modern execution engine without relying on live
data downloads, a synthetic OHLCV sample is bundled at
`backtest/sample_data/sample_ohlcv.csv`.  Run a backtest of the default
SMA-filtered strategy against this dataset with:

```
python scripts/run_backtest.py \
  --source csv --path backtest/sample_data/sample_ohlcv.csv \
  --strategy backtest.strategies.sma_filtered:generate_signals \
  --strategy_args fast=8,slow=34,trend_fast=55,trend_slow=144 \
  --fees_bps 5 --slip_bps 2 --tp_bps 60 --sl_bps 40 \
  --max_bars 18 --notional 1.0 --risk_per_trade 0.01 \
  --out_prefix artifacts/sample_backtest
```

The command writes the trade blotter, equity curve, and metrics beneath
`artifacts/`.  A copy of the latest run is kept under
`backtest/sample_results/` for reference.

### Minimal CSV-only CLI

If you prefer a lightweight wrapper that works with local CSV files only and
skips the more advanced CCXT integration, use the helper script below:

```bash
python scripts/simple_backtest_cli.py \
  --source csv --path backtest/sample_data/sample_ohlcv.csv \
  --strategy_args fast=8 slow=34 trend_fast=55 trend_slow=144 \
  --fees_bps 5 --slip_bps 2 --tp_bps 60 --sl_bps 40 \
  --max_bars 18 --notional 1.0 --risk_per_trade 0.01 \
  --out_prefix artifacts/sample_backtest_simple
```

The script writes the same trio of outputs—blotter, equity curve, and metrics—
to the requested directory.

## Quick SMA Crossover Demo

If you want to experiment with the third-party ``backtesting`` package, run the
standalone moving-average example that ships with the repository:

```bash
python scripts/backtesting_sma_cross_example.py
```

The script prints summary statistics to the console and, when a display backend
is available, pops up an interactive equity-curve plot.


## Trading Readiness Check

Before using the trading bot, run the comprehensive readiness checker:

```bash
python check_trading_readiness.py --verbose
```

To let the checker automatically remedy common issues (install critical
dependencies, create a `.env` file, and seed sample market data), add the
`--fix-issues` flag:

```bash
python check_trading_readiness.py --fix-issues --verbose
```

This validates:
- Environment setup and dependencies
- Configuration and safety settings
- Data availability and quality
- Model validation results
- Risk management settings
- Trading safety mechanisms

For quick setup assistance:

```bash
python quick_setup.py --install-deps
```

See [TRADING_READINESS.md](TRADING_READINESS.md) for detailed information.

## Systematic Trading Principles

For guidance on data hygiene, modeling discipline, validation practices, and production defenses, see the [Systematic Trading Model Principles](docs/systematic_trading_principles.md) guide. It also lists practical thresholds and quick recipes you can adopt immediately.

## Upgrade Roadmap

The prioritized engineering backlog for the next wave of improvements lives in
[docs/upgrade_plan.md](docs/upgrade_plan.md). It captures the enhanced
backtesting, risk-sizing, arbitrage, and reconciler initiatives alongside the
latest repository health check results.

