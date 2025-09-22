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

## Trading Readiness Check

Before using the trading bot, run the comprehensive readiness checker:

```bash
python check_trading_readiness.py --verbose
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
