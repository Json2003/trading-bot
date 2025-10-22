# Phase 5 — Paper Trading Playbook (Weeks 5–6)

This playbook documents the operating checklist, observability requirements, success
criteria, and kill rules for the paper trading dry-run. The objective of this
phase is to validate that the live trading stack behaves identically to the
backtests while running with tiny positions on production infrastructure.

## Markets and Venues

- Start with **Binance spot** markets for:
  - `BTC/USDT`
  - `ETH/USDT`
- Add perpetual futures legs only for hedge coverage after spot execution is
  confirmed to be stable.

## Operational Checklist

1. **Activate the reconciler** with the following behaviors enabled:
   - Idempotent order submission to prevent duplicates.
   - Open-order reconciliation loop.
   - Position sanity loop every 60 seconds.
2. **Configure observability**:
   - Emit structured JSON logs.
   - Forward error and kill events to Telegram/Discord webhooks.
   - Persist equity, beta, and hedge notional snapshots every bar to CSV/JSON.

## Success Criteria (30 Trading Days)

- Tracking error versus backtest is ≤ 15%.
- No duplicate or "ghost" orders and no persistent state desynchronizations.
- Kill-switch must remain inactive under normal market regimes.

## Kill Rules

Trigger the kill-switch and revert to the Phase 2 baseline if **either** of the
following occurs within the first 10 trading days:

- Live Sharpe ratio drops below 0.5.
- Drawdown breaches the configured limits.

Once the kill rules trigger, conduct a full post-mortem, retune the strategy, and
restart from the validated baseline.
