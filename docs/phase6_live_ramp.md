# Phase 6 — Live Ramp (Post-Paper)

Phase 6 governs the transition from paper trading to live capital. The objective is to scale
exposure deliberately, relying on observed live performance and safety telemetry rather than
calendar-based milestones.

## Goal

Scale only when the data demonstrates sufficient edge, risk control, and operational stability.

## Ramp Rules

- **Initial exposure:** Begin with 10–20% of the target position size when exiting the paper-trading phase.
- **Weekly step-ups:** Increase capital deployment in 10% increments each week *only if all promotion gates are met*:
  - Rolling 2-week Sharpe ratio ≥ 1.0.
  - Maximum drawdown remains within the configured limit for the strategy.
  - Execution slippage stays inside the modeled tolerance band.
  - No kill-switch events or other emergency triggers occurred during the evaluation window.
- **Hold steady otherwise:** Freeze scaling (no increase) whenever any of the gates fail; resume evaluation after another full week of compliant performance.

## Emergency Procedures

Trigger an automatic flatten and enter a four-hour cool-off period whenever any of the following conditions occur:

- Intraday drawdown exceeds the configured daily loss limit.
- API latency spikes beyond acceptable thresholds.
- The beta hedge fails to execute on two consecutive attempts.

During the cool-off, halt new order submissions, investigate the root cause, and re-run health checks before resuming even the reduced exposure level.
