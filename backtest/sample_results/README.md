# Sample Backtest Results

This directory captures a reproducible run of the SMA-filtered strategy against
the synthetic hourly OHLCV dataset stored at `../sample_data/sample_ohlcv.csv`.
For lower-latency experiments, a companion one-minute sample lives at
`../sample_data/sample_ohlcv_1m.csv`.

## Summary

- Command: see the "Sample backtest workflow" section in the repository
  `README.md`.
- Trades: 21
- Win rate: 66.67%
- Total return: 9.55%
- Max drawdown: -1.67%
- Sharpe ratio: 2.93
- Sortino ratio: 4.49
- Profit factor: 3.50

Artifacts in this folder:

- `sample_backtest_metrics.json` – core performance statistics.
- `sample_backtest_trades.csv` – trade blotter with exits and PnL.
- `sample_backtest_equity.csv` – equity curve sampled per bar.
