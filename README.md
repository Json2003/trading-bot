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

## Binance to GCS ingestion

Fetch minute and five-minute klines for BTCUSDT and ETHUSDT across spot and
USDT-margined futures markets and upload them to a Google Cloud Storage bucket:

```bash
python tradingbot_ibkr/binance_to_gcs.py --bucket <bucket-name> \
  --symbols BTCUSDT,ETHUSDT --intervals 1m,5m --markets spot,um \
  --start 2024-01-01T00:00:00 --end 2024-01-01T01:00:00
```

Replace the bucket name and time range as needed.
