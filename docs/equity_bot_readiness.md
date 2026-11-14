# NYSE stock research, education and neural execution audit

As of November 14, 2026. Scope: NYSE-listed common stocks; paper research only.

## What the completed cycle establishes

The frozen replay uses IBM, JPM and KO, each verified by provider metadata as
NYSE common equity in USD. It reuses the existing SMA strategy defaults and
its source hash. Each historical block starts with $5,000 cash and has a $1,000
per-position cap. Entries wait for the next session open; whole-share sizing,
prior-session volume limits, 43 bps on each fill, adverse gap stops and a
close-observed drawdown latch are explicit assumptions. The block ledger and
cash equity reconcile. This is a historical diagnostic, not broker paper fills
or unseen confirmation.

The original run and the provenance-corrected run produce the same performance:

| Historical block | Net return | Closed trades | Nonoverlapping observations | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| 2023–2024 | -1.52% | 8 | 7 | -2.19% |
| 2025 | -0.63% | 13 | 10 | -3.50% |
| 2026 through November 13 | -1.56% | 4 | 2 | -2.04% |

All blocks triggered the risk latch and fail the 20-observation minimum. The
2026 latch triggers January 20; most subsequent dates are flat. Losses beyond
two percent are possible because the latch observes close marks and liquidation
waits for the next open. No positive net edge is demonstrated. The fixed cash
buy-and-hold benchmark has different exposure and no latch; it is a reference,
not a matched-risk experiment.

## Concrete skills added on this research branch

- Per-stock SMA 5/20/50/200, RSI 14, MACD 12/26/9, Bollinger 20/2, ATR 14,
  prior-volume comparison and prior support/resistance diagnostics.
- Each output explains the existing buy setup, a signal exit if already holding,
  indicator conflicts and the historical risk latch. Missing history returns
  unknown. The new indicators do not alter the frozen strategy or its results.
- A post-run review attributes execution-cost drag, compares the benchmark,
  counts nonoverlapping observations and identifies operational/data gaps.
- A 16-source learning catalog, including eight books, links concepts to data
  requirements and research translations. Source types distinguish textbooks,
  empirical papers, expert opinion, courses and official rules. Relevant source
  records are attached to each stock report.

The catalog contains original bounded lesson notes grounded in publisher
descriptions/contents, author abstracts/public material and official pages.
It does not contain complete commercial books, does not claim exhaustive
literature coverage, and does not update neural model weights. Model training
and demonstrated trading competence remain separate milestones.

## What the neural execution audit found

1. `scripts/ml_gate.py` returned allow=True on any inference exception and
   defaulted a missing decision to permission. The proposed repair denies
   failures, missing/invalid responses, nonfinite inputs and mismatched tags.
2. `scripts/infer_api.py` reported a caller-requested model tag while continuing
   to use the loaded model. The repair rejects unloaded requested tags and
   reports the actual model/scaler/tag snapshot. Atomic state publication
   prevents a concurrent model reload from mixing those objects. Invalid
   feature shapes/counts, nonfinite values and invalid probabilities are rejected.
3. The API produces a probability and an allow flag. That is a prediction gate,
   not an NYSE execution algorithm. Repository search did not establish a
   validated neural-to-IBKR caller for it. The installed Windows process and
   broker session were not inspected.
4. `tradingbot_ibkr/brokers/order_router.py` validates basic order fields but
   explicitly leaves per-account risk rules as a placeholder. `IbkrBroker`
   delegates to a supplied client; its existence does not establish a connected,
   reconciled broker deployment.
5. `scripts/train_lstm.py` imports the missing `data.feature_store` and fits
   its scaler in `build_sequences` before its later train/validation split.
   This does not establish a leakage-controlled NYSE-trained model. No model
   was trained, activated or promoted here.

The seven neural regression tests use deterministic stub logits. They verify
the actual API/helper boundary behavior, not forecasting performance or a real
broker's execution. The stock cycle has eight execution tests and four
indicator/provenance tests. These checks do not certify the entire trading stack.

## Next capabilities requiring measured evidence

| Capability | Current status | Evidence needed |
| --- | --- | --- |
| Basic indicator calculations | Implemented in research | Locked forward test of incremental net value |
| NYSE eligibility and completed historical bars | Checked for this fixed universe | Point-in-time universe, delistings and session calendar |
| Real execution quality | Assumed in replay | NBBO/auction data, partial fills, queue handling and IBKR reconciliation |
| Earnings/fundamentals | Curriculum only | Filings and earnings release availability, revisions and blackout rules |
| Cash and corporate actions | Partial accrual assumptions | Settled-cash ledger, dividend payment dates and audited split handling |
| Regime/portfolio reasoning | Diagnostic only | Market/sector context, correlation and frozen regime hypotheses |
| Neural-to-stock orders | Not verified | Versioned feature schema, freshness, calibrated model, account risk and end-to-end paper fills |
| Profitable strategy | Not demonstrated | Positive net untouched forward confirmation with sufficient independent observations |

Calendar and halt references: [NYSE hours](https://www.nyse.com/trade/hours-calendars)
and [NYSE trading information](https://www.nyse.com/trade/trading-information).
Settlement reference: [SEC T+1 rule](https://www.sec.gov/rules-regulations/2023/02/34-96930).
These are educational requirements; this branch does not route orders.

## Reproduce

```bash
python -m unittest discover -s tests -p test_nyse_research_cycle.py -v
python -m unittest discover -s tests -p test_equity_analysis.py -v
python -m unittest discover -s tests -p test_neural_execution_integrity.py -v
python scripts/run_nyse_research_cycle.py
```

The read-only NYSE workflow preserves raw responses, hashes, frozen experiment,
education catalog, trade ledger and equity curves. Publication of this branch
does not deploy it to the desktop, change active model tags, or authorize orders.
