# Cryptocurrency research platform

This directory defines the research contract for the bot. A hypothesis is a
falsifiable, versioned rule with a named data dependency, causal timestamp,
entry/exit specification, cost model, and validation status.

## Lifecycle

1. Register a hypothesis in `hypotheses.yaml`.
2. Run a discovery scan on completed data only.
3. Classify the result: promising, cost-sensitive, regime-specific, unstable,
   insufficient-sample, or rejected.
4. Freeze a bounded variant before validation.
5. Evaluate once on the untouched validation segment.
6. Evaluate future data only after the immutable checkpoint advances.
7. Require independent forward evidence before confirmation.

Historical repeats are reproducibility checks, not new trials. Overlapping
rolling windows are not independent evidence. A negative result is retained
with its failure reason instead of silently removing the hypothesis.

## Non-negotiable boundaries

The research cycle cannot place orders, enable leverage, change live risk
limits, or promote a strategy. Every report must include the data fingerprint,
cutoff, cost assumptions, trades, net return, drawdown, Sharpe, profit factor,
and checkpoint transition.

The registry is deliberately broad but finite. “All hypotheses” means the
predeclared measurable families, not unlimited chart-pattern mining.
