# Shock-continuation confirmation preregistration

This is one new, frozen, research-only hypothesis. It is not a tuning revision
to PR #253 or PR #254. The rule, confirmation condition, dates, horizon, and
gates are fixed before reviewing results.

## Frozen rule

At completed hourly bar **t**, independently for BTCUSDT and ETHUSDT:

- log return <= -2 prior 30-day hourly standard deviations;
- volume >= 1.5 times the prior 24-hour median;
- candle range is at or above the 75th percentile of prior 24-hour ranges;
- no moving-average filter is used;
- the next completed candle (**t+1**) must close below its open and at or below
  the midpoint of the shock candle;
- enter short at the open of **t+2**, representing next-bar entry plus one-hour
  latency;
- exit after six hours;
- apply a 12-hour per-asset cooldown.

The next-candle condition is confirmation, not holdout selection. There are no
alternate horizons, thresholds, assets, stops, targets, or parameter grids.

## Fixed evaluation

- Full window: 2023-01-01 through 2026-07-31 UTC.
- Discovery: 2023-01-01 through 2025-03-31.
- Untouched confirmation: 2025-04-01 through 2026-07-31.
- Six chronological blocks per segment; at least 20 trades per block.
- Every block must have positive net return.
- Median block return must cover at least one shared stress round-trip cost.
- Net P&L uses the shared stress model, latency, partial fills, rejection,
  funding, and execution costs.

This remains research-only. It places no orders, enables no leverage, and cannot
promote a strategy.
