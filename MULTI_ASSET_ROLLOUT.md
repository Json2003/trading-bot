# Multi-Asset Rollout Policy

This infrastructure defines the account-growth milestone without turning it into an
automatic live-trading switch.

## Operating rules

- The policy is permanently `paper_research_only`; live orders are not authorized.
- Global leverage remains disabled.
- The $25,000 trigger uses settled equity. Crossing it makes sleeves eligible for
  paper validation only.
- A $5,000 buffer separates the milestone from the operating floor. The evaluator
  reports the buffer, but does not authorize live trading.
- Each sleeve has independent capital and risk limits. There is no cross-margining
  or shared unbounded risk pool.
- Options are defined-risk only; futures are micro-contract research only; forex
  leverage is disabled.
- Promotion requires positive net performance after costs, positive median
  walk-forward performance, stress drawdown compliance, at least 30 trades, no
  leakage, an exact fill ledger, and explicit human approval.

## Usage

```bash
python scripts/evaluate_multi_asset_rollout.py --equity 25000
python -m unittest -v tests/test_multi_asset_rollout.py
```

The output is an auditable JSON decision. It never returns a live-activation
authorization. Research runners should consume this decision before creating
any instrument-specific paper experiment.

## Required next adapters

Before any paper sleeve is considered for promotion, add and test:

1. stock/ETF corporate-action and borrow-cost handling;
2. option chain, Greeks, implied-volatility, assignment, and max-loss ledger;
3. futures contract specifications, expiry/roll, margin, and liquidation model;
4. forex spread, rollover, session, and currency-conversion model;
5. broker fill reconciliation against the immutable execution ledger.

The milestone is therefore a workflow gate, not a claim that a strategy is
profitable or that a broker/regulator permits a particular trading pattern.
