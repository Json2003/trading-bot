# "Outthink the Market" Enhancements

These initiatives can be slotted in once the core trading system is stable. They layer additional alpha generation, risk awareness, and adaptive tuning on top of the existing momentum-driven framework.

## Order-Book Microstructure Overlay
- Capture features such as quote imbalance, replenishment times, and cancellation bursts to refine momentum entries with a micro-alpha signal.
- Prioritize execution when microstructure flow supports the primary directional bias; back off when order-book pressure contradicts the signal.

## Liquidation Heatmap Throttling
- Integrate perpetuals liquidation data to build a heatmap that highlights imminent squeeze zones.
- Down-weight or delay entries when liquidation risk becomes extreme to avoid getting caught in forced unwind cascades.

## Entropy-Based Position Sizing Gate
- Track prediction entropy (or similar confidence measures) from the model ensemble.
- Scale positions down by roughly 50–80% when entropy rises, signaling market indecision.

## Regime-Segmented Nightly Optuna Runs
- Re-run Optuna hyperparameter searches each night across datasets segmented into bull, bear, and choppy regimes.
- Persist the best regime-specific parameter sets and load them dynamically based on live regime detection.
