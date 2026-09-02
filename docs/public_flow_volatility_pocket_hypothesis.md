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

The following values are fixed before examining the research outcome:

1. For each symbol, calculate rolling reference distributions from the prior
   24 hours of completed one-minute rows only.
2. A volatility pocket requires five-minute realized absolute return at or above
   the prior-24-hour 90th percentile.
3. A volume pocket requires total aggressive notional
   (buy_notional + sell_notional) at or above the prior-24-hour 95th
   percentile.
4. Flow direction requires
   net_aggressive_notional / (buy_notional + sell_notional) to be at least
   +0.30 for a long or at most -0.30 for a short.
5. The latest completed book imbalance must agree with the flow direction:
   at least +0.10 for a long or at most -0.10 for a short.
6. Enter at the next completed-minute midpoint, with one-minute latency.
7. Exit after a fixed 30 completed minutes. Do not use future candles to alter
   the exit, threshold, or direction.
8. Do not open another position until the prior position and its 30-minute
   cooldown are complete. Signals inside an active window are not separate
   evidence.
9. Use fixed trade notional, the existing 86-basis-point round-trip stress
   execution model, and the existing partial-fill/rejection assumptions.

Only rows marked completed=true are eligible. Missing fields, gaps, overlaps,
and final partial minutes are excluded or marked unknown; they are never
treated as zero.

## Evaluation design

The first immutable public-flow window must be at least 90 continuous days for
BTCUSDT and ETHUSDT, with no gap or overlap in its checkpoint history. Split
chronologically into six non-overlapping 15-day blocks. The first four blocks
are development; the final two are untouched confirmation data.

All thresholds and the 30-minute hold are frozen before the first development
result is inspected. The confirmation blocks remain unavailable to discovery,
diagnostic selection, or tuning. A repeated run on the same checkpoint is a
deterministic reproduction, not new evidence.

Report separately for each symbol, each development/confirmation segment, and
each six-block result:

- net return and net P&L
- maximum drawdown
- Sharpe proxy
- profit factor
- trade count and win rate
- gross P&L and modeled execution costs
- data-through timestamp, gap/overlap counts, and excluded/unknown rows

The existing confirmation gates remain in force. A positive development result
alone cannot advance the candidate. A candidate advances only if it passes the
untouched confirmation and sample/block requirements under the same stressed
cost model.

## Safety boundary

No live or paper orders, leverage, risk-limit changes, broker credentials, or
automatic promotion are permitted. A negative or insufficient result is
reported as research evidence and does not trigger parameter changes.
