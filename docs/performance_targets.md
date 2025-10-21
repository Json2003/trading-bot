# Portfolio Performance Targets

The trading program tracks a fixed set of live-readiness metrics to ensure strategies and infrastructure meet the risk committee's guardrails.  The table below lists the current thresholds.

| Metric | Target | Notes |
| --- | --- | --- |
| Sharpe (out-of-sample) | ≥ 1.0 | Evaluated on the most recent out-of-sample window after accounting for realistic costs. |
| Sortino | ≥ 1.3 | Downside deviation uses the same cost assumptions as the Sharpe calculation. |
| Max Drawdown | ≤ 25% (spot), ≤ 30% (with perps) | Hard stop for realised drawdown from the running equity high; tighter 25% cap applies to unhedged spot-only deployments. |
| Profit Factor | ≥ 1.2 | Ratio of gross wins to gross losses over the monitored period. |
| Win-rate | ≥ 52% (strategy-dependent) | Applies at the portfolio level; individual strategies may run lower if they materially improve the aggregate Sharpe. |
| Turnover cost | ≤ 0.3 × gross alpha | Estimated annualised trading costs divided by pre-cost returns. |
| Hedge efficiency | Beta within 0.1–0.2 target 80% of time | Rolling 60-minute beta of the hedged sleeve versus its benchmark should sit inside the band at least 80% of the time. |
| Uptime (paper/live) | ≥ 99% market hours | Combined availability for paper and live deployments during market trading hours. |
| Incident MTTR | < 5 m | Median time to resolve incidents that breach guardrails or disable trading. |

Reassess these limits quarterly or whenever market conditions or mandate changes warrant, and document any overrides with a clear expiry date.
