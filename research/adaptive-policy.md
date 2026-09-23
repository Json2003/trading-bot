# Governed adaptive strategy policy

The adaptive layer may select among pre-registered candidates, but it may not invent rules during a run or learn from the confirmation segment.

## Decision process

1. Build the regime label from information available before the decision candle closes.
2. Look up the candidate set registered for that regime.
3. Select the candidate using only prior completed evidence and a fixed tie-break rule.
4. Require expected net movement to exceed the frozen stress-cost hurdle.
5. Trade only when the candidate and data-quality gates pass.
6. Record the regime, candidate, evidence window, and reason for every decision.

## Allowed regime labels

- trend_up
- trend_down
- range
- volatility_expansion
- volatility_contraction
- panic
- recovery
- unknown

## Safety rules

- Unknown or stale regime means stand aside.
- Candidate selection never uses future returns.
- A strategy cannot be switched because the current trade is losing.
- Repeated rolling windows are not independent confirmation.
- No leverage, orders, risk-limit changes, or automatic promotion are permitted.
- A candidate must survive its own discovery, validation, and genuinely new forward checkpoint.

The adaptive policy is an orchestration layer. It does not change the frozen signal definition of any candidate.
