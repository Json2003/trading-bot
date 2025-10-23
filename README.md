# Splitstar Operations Console
Splitstar Operations Console (formerly the trading-bot project) is the
operations and research surface for Splitstar's agent-based trading stack.
It keeps the `tradingbot_ibkr` package for backwards-compatible scripts while
rebranding the product experience, dashboards, and deployment targets under the
Splitstar Operations umbrella.

> 💡 **Repository rename:** clone the repo into `splitstar-operations-console`
> (or update existing checkouts) to match the new product branding. Scripts
> continue to recognise the legacy `trading-bot` directory for compatibility.

## Desktop installation

Run one of the provided scripts to set up a virtual environment and install
dependencies.

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

The console supports multiple asset classes including forex, options, futures,
crypto, and stocks via a unified `AssetClass` enum. Trading scripts and the
engine can adjust risk settings based on the selected class.

## Portfolio performance guardrails

Target operating metrics for the live portfolio—covering risk-adjusted returns,
drawdown limits, turnover costs, hedging quality, and operational readiness—are
documented in [`docs/performance_targets.md`](docs/performance_targets.md).
Review the table before promoting new strategies or adjusting allocations so
changes stay within the agreed guardrails.

## Binance to GCS ingestion

Fetch minute and five-minute klines for BTCUSDT and ETHUSDT across spot and
USDT-margined futures markets and upload them to a Google Cloud Storage bucket:

```bash
python tradingbot_ibkr/binance_to_gcs.py --bucket <bucket-name> \
  --symbols BTCUSDT,ETHUSDT --intervals 1m,5m --markets spot,um \
  --start 2024-01-01T00:00:00 --end 2024-01-01T01:00:00
```

Replace the bucket name and time range as needed.
