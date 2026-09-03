# Frozen hypothesis: public-flow confirmed volatility pocket continuation

## Status

This is a paper-only research specification. It is not an executable trading
profile and has no account, broker, order, leverage, risk-setting, or promotion
path.

## Research question

When a completed one-minute market-flow window shows an unusually large burst
of traded notional and realized movement, does the combination of aggressive
buy/sell imbalance and top-of-book imbalance predict the next short-horizon
move after modeled execution costs?

This is a new data-source test using public aggregate trades, bookTicker
snapshots, and completed-minute summaries. It is not a rerun of the prior
candle-only breakout/reversal rules.

## One frozen rule

The values below are fixed before examining any outcome:

1. For each symbol, calculate rolling reference distributions from the prior
   24 hours of completed one-minute rows only.
2. A volatility pocket requires five-minute realized absolute return at or above
   the prior-24-hour 90th percentile.
3. A volume pocket requires total aggressive notional
   (buy_notional + sell_notional) at or above the prior-24-hour 95th
   percentile.
4. Flow direction requires net aggressive notional divided by total notional to
   be at least +0.30 for a long or at most -0.30 for a short.
5. The latest completed book imbalance must agree with direction: at least +0.10
   for a long or at most -0.10 for a short.
6. Enter at the next completed-minute midpoint with one-minute latency.
7. Exit after a fixed 30 completed minutes.
8. Do not open another position until the prior position and its 30-minute
   cooldown are complete. Signals inside an active window are not separate
   evidence.
9. Use fixed trade notional, the existing 86-basis-point round-trip stress
   execution model, and the existing partial-fill/rejection assumptions.

Only completed rows are eligible. Missing fields, gaps, overlaps, and final
partial minutes are excluded or marked unknown; they are never treated as zero.

## Staged evaluation design

The three-minute sample and six-hour collection are pipeline checks only.

The first meaningful screen uses a continuous 60-day BTCUSDT/ETHUSDT archive:
the first two days provide rolling-feature warmup, the next 30 days are
development data, and the final 28 days are untouched confirmation data. This
is an early evidence screen, not a final year-long confirmation.

The original one-year protocol remains the final standard: six months of
development followed by six months of untouched confirmation when enough
archived public-flow history exists.

All thresholds and the 30-minute hold are frozen before results are inspected.
The confirmation data cannot be used for discovery, diagnostics, or tuning. A
repeat on the same checkpoint is deterministic reproduction, not new evidence.

Report by symbol, segment, and block:

- net return and net P&L
- maximum drawdown
- Sharpe proxy
- profit factor
- trade count and win rate
- gross P&L and modeled execution costs
- data-through timestamp, gap/overlap counts, and excluded/unknown rows

A positive development screen alone cannot advance the candidate. Confirmation
still requires the existing sample, block, and stressed-cost gates.

## Safety boundary

No live or paper orders, leverage, risk-limit changes, broker credentials, or
automatic promotion are permitted. A negative or insufficient result is
reported as research evidence and does not trigger parameter changes.
