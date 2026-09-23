# Bounded v3 development screen

`scripts/run_v3_exploration.py` is a research-only screening tool for the
adaptive BTC/ETH momentum-volatility model. It is separate from the continuous
controller and cannot mark a strategy ready, alter the active paper profile,
enable leverage, or place an order.

## What it evaluates

The suite contains exactly 12 named v3 configurations: the existing balanced,
selective, and conservative variants plus nine pre-declared hypotheses for
trend speed, volatility expansion, entry quality, leader selection, liquidity,
edge, and exit handling. The source-defined suite is intentionally bounded;
adding or removing candidates is a new experiment, not a continuation of the
same evidence.

Every candidate is evaluated at both $4,000 and $6,000 order notionals under
the normal 10-bps fee plus 5-bps slippage costs and the higher 20-bps fee plus
10-bps slippage stress costs. A development screen pass requires both sizes,
positive base and stress development returns, positive median returns across
three non-overlapping development folds, minimum entry counts, and no permanent
halt or repeated kill-switch event.

## Protected confirmation year

The latest complete one-year period is deliberately excluded from every screen
calculation and report. A development-screen pass is only a manually reviewed
shortlist for a later frozen confirmation run; it is never a promotion result.
The controller does not count rolling historical data vintages as independent
confirmation or set `strategy_ready`; confirmation must be manually frozen and
use new, non-overlapping future data.

## Manual run

From GitHub Actions, choose **V3 Bounded Development Screen**, select the
research branch containing the screen, and use **Run workflow**. The workflow
downloads completed monthly Binance Vision data, uploads its report as an
artifact, and has read-only repository permissions.
