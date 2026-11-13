# Liquidity-shock mean-reversion preregistration

This is one research-only experiment. The rule, dates, horizon, and gates are
frozen before reviewing results. It does not place orders, enable leverage, or
promote a strategy.

## Frozen hypothesis

A high-volume downside liquidity shock in BTC or ETH, followed while price
remains above its 200-hour EMA, partially mean-reverts during the next six
hours.

Signal at completed hourly bar t:

- log return at t is at or below -2 prior-sample standard deviations;
- volume is at least 1.5 times the prior 24-hour median;
- the candle's range percentile versus the prior 24 hours is at or below 25%;
- close is above the 200-hour EMA.

Entry is the open of bar t+2: next-bar entry plus one-hour latency. Exit is
the close six hours later. There is no stop, target, alternate horizon, or
parameter search. The same rule is applied independently to BTCUSDT and
ETHUSDT, with no leader selection. A 12-hour per-asset cooldown prevents
overlapping re-entry.

## Fixed sample

- Full window: 2023-01-01 00:00 UTC through 2026-07-31 23:00 UTC.
- Discovery: 2023-01-01 through 2025-03-31.
- Untouched confirmation: 2025-04-01 through 2026-07-31.

## Gates

Every segment is split into six chronological blocks. Each block must contain
at least 20 trades, have positive mean net return, and the segment's median
block return must cover at least one shared stress round-trip cost. The
confirmation period is never used to select or alter the rule.

Costs, latency, partial fills, funding, and rejected fills come from
scripts/execution_model.py. A passing result is still research evidence only;
it is not permission for paper or live deployment.
