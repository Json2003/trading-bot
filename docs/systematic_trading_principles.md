# Systematic Trading Model Principles

This guide captures the core practices we follow when building and operating systematic trading models. Apply these guardrails end to end—from raw data to live deployment—to keep leakage out, avoid overfitting, and respond quickly when market regimes change.

## 1. Data Hygiene – Kill Leakage Early
- **Respect time order.** Work strictly in chronological order and never shuffle samples during preprocessing or modeling.
- **Use purging and embargo.** When creating time-based splits, remove training rows whose label horizons overlap with the validation or test fold and add an embargo at least equal to the label horizon after every test window.
- **Lag alternative data.** Apply realistic publication lags plus a conservative buffer to any non-price data (fundamentals, macro, alternative data) before using it as a feature.
- **Avoid survivorship and look-ahead bias.** Rely on point-in-time universes, fundamentals, and membership lists; freeze each dataset by reference date.
- **Model costs directly.** Bake trading costs (fees, slippage) into the training targets or loss so the model optimizes net returns.

## 2. Modeling Discipline – Favor Stability Over Cleverness
- **Start simple.** Establish a strong tabular baseline (e.g., XGBoost or LightGBM) before experimenting with sequence models.
- **Regularize aggressively.** Use shallow trees (max depth ≤ 4), strong L1/L2 penalties, feature and row subsampling ≤ 0.8, and early stopping.
- **Keep features sane.** Limit the feature set to low-collinearity, stationary transforms (log returns, z-scores). Monitor rolling means and variances; drop features whose distributions drift excessively.
- **Test for spurious edge.** Confirm the model’s performance collapses when the target is permuted and enforce monotonic constraints where economically justified.
- **Ensemble across regimes.** Average models trained on different time blocks to improve robustness rather than relying solely on different random seeds.

## 3. Brutal Validation – Make It Earn Every Basis Point
- **Use walk-forward validation.** Prefer purged, embargoed time-series cross-validation with 5–10 folds. Tune hyperparameters only inside the folds; never on the final test set.
- **Adopt nested tuning.** Use outer folds for scoring and inner folds for hyperparameter selection, or restrict search spaces to conservative ranges.
- **Test across regimes.** Maintain multiple regime-aware test sets (bull, bear, choppy, crisis) and require the strategy to succeed in more than 60% of them.
- **Quantify uncertainty.** Apply block bootstraps (1–3 month blocks) to estimate Sharpe, CAGR, and max drawdown distributions.
- **Control multiple testing.** Deflate Sharpe ratios (e.g., Deflated Sharpe, Probabilistic Sharpe) to adjust for idea fishing and accept only persistent effects.
- **Evaluate turnover and capacity.** Ensure the signal remains profitable after realistic costs at intended trade sizes.
- **Run sanity checks.** Verify shuffled-return predictions produce Sharpe ≈ 0, one-bar execution delays remain profitable, and no-trade days (confidence thresholds) improve risk versus always-on trading.

## 4. Production Defenses – Manage Inevitable Edge Decay
- **Champion–challenger workflow.** Promote new models only after shadowing them for 4–8 weeks against a stable champion.
- **Monitor for drift.** Track population stability index (PSI), Kullback–Leibler divergence, rolling AUC/IC, and change-point tests (CUSUM, ADWIN).
- **Set guardrails.** Enforce daily loss limits, leverage caps, per-asset position limits, and kill switches triggered by drift or VaR/drawdown breaches.
- **Retrain deliberately.** Retrain on rolling 2–3 year windows but promote only when the new model improves out-of-sample Sharpe with overlapping confidence intervals, keeps drawdown/turnover in check, and exhibits stable feature importance or SHAP profiles across folds.
- **Adjust exposure to decay.** Scale positions down when live IC, hit rate, or t-stat fall below thresholds; restore exposure only after sustained recovery over N trades.
- **Perform post-trade analytics.** Attribute P&L by feature, sector, and liquidity bucket and remove persistently underperforming segments.

## Practical Thresholds
- **Embargo:** 1× the label horizon (e.g., 1 day for next-day forecasts, 1 week for 5-day forecasts).
- **Promotion gates:** Live IC ≥ 0.02–0.04 or hit rate ≥ 52–55% (net of costs) across at least 200 trades.
- **Risk caps:** Daily loss limit of 1–2× expected daily volatility; drawdown alert at 10–15% with a hard stop at 20–25%.
- **Turnover:** Target annual turnover between 50 and 200 depending on cost and liquidity assumptions.

## Quick Recipes to Implement Now
- **Prevent leakage.** Rebuild features using only data available at time *t-1* and assert all feature timestamps precede label timestamps.
- **Use confidence thresholds.** Take trades only when the probability of an upward move exceeds 0.55 (or when in the top-*k* ranks) to boost Sharpe and reduce churn.
- **Add regime gates.** Disable or resize exposure when risk metrics (e.g., VIX) exceed limits or when trend filters (e.g., price below 200-day moving average) turn negative.
- **Check stability.** Compare results across staggered train/test windows (e.g., train 2013–2017, test 2018–2019; train 2015–2019, test 2020–2021) and expect performance metrics to be consistent.

