# Multi-Asset Rollout and Capital Growth Policy

This infrastructure defines the account-growth milestone ladder without turning
any milestone into an automatic live-trading switch.

## Milestone ladder

`$25,000 → $50,000 → $100,000 → $250,000 → $500,000 → $1,000,000`

The planner uses settled equity and reports the next target, dollar gap,
contribution-only months, and an optional reconciliation of starting equity,
deposits, and verified net P&L. It deliberately does not forecast returns.

## Operating rules

- The policy is permanently `paper_research_only`; live orders are not authorized.
- Global leverage remains disabled.
- Reaching $25,000 makes sleeves eligible for paper validation only.
- Each higher milestone is a capital-accounting checkpoint, not permission to
  increase risk.
- A $5,000 buffer separates the first milestone from the operating floor.
- Each sleeve has independent capital and risk limits. There is no cross-margining
  or shared unbounded risk pool.
- Options are defined-risk only; futures are micro-contract research only; forex
  leverage is disabled.
- Promotion requires positive net performance after costs, positive median
  walk-forward performance, stress drawdown compliance, at least 30 trades, no
  leakage, an exact fill ledger, and explicit human approval.

## Usage

```bash
python scripts/evaluate_multi_asset_rollout.py \
  --equity 15000 \
  --monthly-contribution 500 \
  --starting-equity 5000 \
  --net-contributions 10000 \
  --verified-net-pnl 0

python -m unittest -v tests/test_multi_asset_rollout.py
```

The output is an auditable JSON decision. It never returns live-activation
authorization.

## Required adapters before any promotion

1. stock/ETF corporate-action and borrow-cost handling;
2. option chain, Greeks, implied-volatility, assignment, and max-loss ledger;
3. futures contract specifications, expiry/roll, margin, and liquidation model;
4. forex spread, rollover, session, and currency-conversion model;
5. broker fill reconciliation against the immutable execution ledger.

The ladder is therefore a planning and governance framework, not a claim that a
strategy is profitable or that any broker/regulator permits a particular pattern.
