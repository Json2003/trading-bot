# Range-expansion mean-reversion preregistration

This is a new single research hypothesis. It is not a tuning revision to the
prior zero-trade result. The frozen rule, dates, horizon, and gates are fixed
before this run. It places no orders, enables no leverage, and cannot promote
a strategy.

## Frozen rule

At completed hourly bar t, independently for BTCUSDT and ETHUSDT:

- log return <= -2 prior 30-day hourly standard deviations;
- volume >= 1.5 times the prior 24-hour median;
- current candle range is at or above the 75th percentile of prior 24-hour ranges;
- close is above the 200-hour EMA.

Enter at t+2 open, representing next-bar entry plus one-hour latency. Exit at
the close six hours later. Apply a 12-hour per-asset cooldown. There are no
alternate horizons, thresholds, stops, targets, asset selection rules, or
holdout-driven changes.

## Fixed evaluation

- Full window: 2023-01-01 through 2026-07-31 UTC.
- Discovery: 2023-01-01 through 2025-03-31.
- Untouched confirmation: 2025-04-01 through 2026-07-31.
- Six blocks per segment; at least 20 trades per block.
- Every block must have positive net return.
- Median block return must cover at least one shared stress round-trip cost.
- Net P&L includes the shared stress model, latency, partial fills, rejection,
  funding, and execution costs.

## Interpretation

A zero-trade result is an implementation/data-coverage failure, not a
profitability failure. A nonzero but losing holdout is evidence against this
specific OHLCV hypothesis, not against all possible market edges.
