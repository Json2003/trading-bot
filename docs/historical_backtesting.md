# Historical backtesting workflow

The paper research workflow uses official Binance Vision kline archives and
does not require exchange credentials. The downloader prefers monthly archives,
falls back to daily archives when a month is not published, verifies the
official SHA256 checksum, and writes normalized UTC OHLCV CSVs.

The source archive layout and timestamp details are maintained in the
[Binance Public Data repository](https://github.com/binance/binance-public-data).

## Download real BTC/ETH/SOL history

Fetch enough warm-up history before the scored year. Hourly bars are the
default because they make the one-day, one-week, one-month, and one-year
horizons explicit while keeping the research run bounded:

```bash
python scripts/fetch_binance_klines.py \
  --market spot \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2025-02-01 \
  --raw-root data/raw/binance-vision \
  --output-root data/historical/binance
```

The downloader writes one normalized file per symbol under the output root and
a manifest containing archive URLs, row counts, and checksums. A missing
historical range is reported as an error; rows are never synthesized.

## Run the four horizons

```bash
python scripts/run_historical_backtests.py \
  --data-root data/historical/binance \
  --interval 1h \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --output var/backtests/historical.json
```

The primary results are the causal momentum/volatility-regime variants. The
ADX/ATR trend strategy is a diagnostic comparator. Each horizon gets its own
flat starting point and uses only the history before that horizon for indicator
warm-up. Results with missing coverage or insufficient warm-up are explicitly
marked `insufficient_data`.

The execution assumptions in the default report are 10 bps fees and 8 bps
slippage per fill, 1.5 ATR stop, 3 ATR target, 0.5% risk per trade, 90% maximum
notional fraction, shorting enabled, and a 24-bar maximum hold. These are
paper assumptions, not a live-trading recommendation.
