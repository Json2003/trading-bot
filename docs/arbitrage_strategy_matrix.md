# Arbitrage strategy matrix

The project should evaluate arbitrage variants independently in paper mode:

1. Pre-funded cross-exchange spot: primary candidate; no transfer during execution.
2. Same-venue triangular: secondary candidate; reject unless all three legs clear costs.
3. Spot/perpetual basis: research-only until funding, margin, liquidation, and borrow costs are modeled.
4. Statistical pairs: research-only; it is convergence trading, not risk-free arbitrage.
5. Latency arbitrage: out of scope for this project until professional market-data and execution infrastructure exists.

Do not combine these into one live allocation. Every candidate needs its own data, cost model, fills, P&L, drawdown, and paper-probation record.
