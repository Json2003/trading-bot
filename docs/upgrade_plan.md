# Trading Bot Upgrade Roadmap

This document captures the immediate upgrade priorities for the trading bot, along with
supporting notes from the latest repository health check.

## Repository Health Check

- **Automated tests:** `pytest`
- **Result:** 108 passing tests (3 warnings) as of 2025-10-13 16:31:31 UTC.
- **Notes:** No test failures observed. Warnings stem from existing test fixtures and do not
  block execution.

## Upgrade Priorities

### 1. Replace Toy Backtest Fills with Full Backtesting Engine

- Integrate the production-grade order management logic (market/limit orders, partial fills,
  ATR-driven stop management) into the backtest module.
- Model slippage realistically by applying liquidity curves and spread-based adjustments.
- Include funding and trading-fee models for derivatives and spot venues, respectively.
- Ensure results remain reproducible through deterministic seeding and configuration
  snapshots.

### 2. Introduce Per-Strategy Risk Sizing

- Compute position size from the configured per-trade risk percentage and the ATR-based stop
  distance for each strategy.
- Enforce min/max position sizing constraints per market to keep exposure within acceptable
  bounds.
- Extend configuration to support volatility-scaling parameters and per-market overrides.

### 3. Implement Arbitrage Leg Management

- Operate pre-funded legs on both exchanges to guarantee simultaneous execution when spreads
  appear.
- Add imbalance handling: if one leg fails, immediately hedge via available liquidity or
  flatten exposure using the reconciler.
- Track fill confirmations and latency per venue to continuously refine the hedge logic.

### 4. Wire Reconciler Kill-Switch to Portfolio Equity Curve

- Monitor portfolio equity in real-time and trigger the kill-switch when configured drawdown
  thresholds are breached.
- Emit structured alerts through Discord and Telegram webhooks for transparency and
  incident response.
- Record kill-switch activations and alert acknowledgements for post-mortem analysis.

## Optional Enhancements

- Integrate a portfolio backtest reporter that computes Sharpe, Sortino, Max Drawdown, and
  Profit Factor metrics. Persist reports with metadata (Git SHA, random seeds, fee
  assumptions) for downstream PnL aggregation.
- Continue repository hygiene: prune unused artifacts, consolidate duplicated configuration
  files, and document any manual steps required for deployment.

