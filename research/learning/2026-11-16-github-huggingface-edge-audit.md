# NYSE edge research learning record — 2026-11-16

## Outcome and evidence status

No inspected project establishes a verified executable edge for this bot's NYSE-only, cash-only, whole-share and 86-bps round-trip stress-cost contract. The leading new research hypothesis is a low-turnover stock-ranking strategy with a retention buffer. This is a source/code audit and proposed experiment, not a completed market-data reproduction, model-training run or strategy promotion.

Author-reported returns below were not independently rerun. No model weights or foreign trading code were installed or executed. No brokerage was connected. Existing scheduled workflows, secrets, cloud resources, active models, risk controls and trading gates were not changed.

The prior audit remains available on [NYSE PR #290](https://github.com/Json2003/trading-bot/pull/290), under research/strategy_sources/2026-11-14-external-edge-audit.md at commit 236cf1cc41b01fc2edcdec901244c766ebbf4aa9. That PR was still draft/unmerged when inspected. The hourly crypto collector pause in [PR #291](https://github.com/Json2003/trading-bot/pull/291) remains intact.

## Candidate queue

| Candidate | Evidence reviewed | Decision |
| --- | --- | --- |
| Buffered stock rankings | Decision-layer source, protocol, numerical summaries, limitations and provenance | First research priority; trading-efficiency hypothesis |
| Low-turnover equity momentum | Features described by author; actual cost accounting and configuration inspected | Useful baseline; reported performance not transferable |
| Defensive multi-factor stock ranking | CRSP/Compustat methodology and strategy source | Later candidate; data and accounting audit required |
| Kronos-small | Original Hugging Face model card and prior audit | Forecast benchmark only |
| FinCast | Original Hugging Face card, official code README and paper abstract | Probabilistic forecast benchmark only |
| LearnFinance SAC | Current original model card | Insufficient published evaluation detail for an edge claim |

## 1. Best new lead: retention buffers

[Original project](https://github.com/thamhienhoai-max/stockmixer-buffered-portfolio/tree/b4300af2117ab9e81b88dbcd8f991e5d672a2a7f) inspected at b4300af2117ab9e81b88dbcd8f991e5d672a2a7f.

The method initially selects five stocks, keeps an existing holding while it remains in the top ten, and fills vacancies from the best remaining ranks. Its [implementation](https://github.com/thamhienhoai-max/stockmixer-buffered-portfolio/blob/b4300af2117ab9e81b88dbcd8f991e5d672a2a7f/src/portfolio/buffered.py) makes retention deterministic; it does not improve the predictor itself.

The author's [cost table](https://github.com/thamhienhoai-max/stockmixer-buffered-portfolio/blob/b4300af2117ab9e81b88dbcd8f991e5d672a2a7f/results/cost_sensitivity.csv) reports mean net Sharpe rising from 0.965 to 1.389 under its 10-bps convention and from 0.789 to 1.317 under 20 bps. These conventions must be mapped to actual traded notional before comparing them with this bot's 43-bps-per-side stress model.

The [limitations](https://github.com/thamhienhoai-max/stockmixer-buffered-portfolio/blob/b4300af2117ab9e81b88dbcd8f991e5d672a2a7f/docs/LIMITATIONS.md) identify a selected 474-stock historical universe, same-close primary timing and prior examination of the test period. The [provenance document](https://github.com/thamhienhoai-max/stockmixer-buffered-portfolio/blob/b4300af2117ab9e81b88dbcd8f991e5d672a2a7f/docs/REPRODUCIBILITY.md) says full datasets, predictions and checkpoints are excluded. Public consistency checks cannot independently regenerate the investment result.

Inference: retaining a still-competitive holding could reduce friction. It cannot rescue a predictor with no useful signal. Borrow the hypothesis, not the published profitability label.

## 2. Equity momentum as the baseline

[Quant Trader](https://github.com/mcscott2001/quant-trader/tree/28e1970331890c0ebf18dbf1fde8f5820ebe95c1) inspected at 28e1970331890c0ebf18dbf1fde8f5820ebe95c1.

The author reports momentum CAGR of 19.6% versus 13.9% for SPY during 2018–October 2026, with 31.4% maximum drawdown and 20-bps-per-side costs. Its ML combination performs worse than momentum. These remain author results on a present-day 66-stock surviving large-cap universe containing both NYSE and Nasdaq names.

The [backtester](https://github.com/mcscott2001/quant-trader/blob/28e1970331890c0ebf18dbf1fde8f5820ebe95c1/quantbot/backtest.py) delays target execution, charges changes in weights and models drift. The [configuration](https://github.com/mcscott2001/quant-trader/blob/28e1970331890c0ebf18dbf1fde8f5820ebe95c1/config.yaml) is weekly rebalancing with an ML live-strategy selection, while the README describes a move to monthly momentum. A claimed current deployment therefore cannot be inferred from the checked-in settings. Its capital, fractional-share and universe assumptions also differ from ours.

Inference: test a simple lower-turnover ranking rule before adding larger models. Do not transfer its quoted CAGR or broker module into this bot.

## 3. Multi-factor alternative: useful ideas, unresolved accounting

[US equity multi-factor project](https://github.com/EricYang16/Cross-Sectional-Momentum-US-Equity) reports 16.30% CAGR and 1.18 Sharpe for a 1995–2024 long-only simulation at five bps per side. Raw CRSP/Compustat data are licensed and excluded. The universe includes NYSE, AMEX and Nasdaq; it is not our NYSE-only contract.

Direct inspection of src/04_run_multifactor_strategy.py found two issues requiring investigation before reproduction:
- Eligibility filters out rows with missing next_month_ret before selecting holdings. That uses knowledge of future return availability and needs a point-in-time/delisting audit.
- Costs use changes in unscaled portfolio weights multiplied by the current regime exposure. If target names/weights stay fixed but exposure changes, that exposure trade is not represented in this turnover calculation. Drift and actual traded-notional accounting need reconstruction.

These are code-audit observations, not a quantified correction to the author's returns. The source is a mechanism reference, not a trusted result.

## 4. Hugging Face findings

[Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) provides financial candlestick forecasts with an official 24.7M-parameter checkpoint. The existing November 14 audit already identifies that its cited investment experiment does not establish NYSE execution under this bot's costs. Any later test needs a pinned checkpoint, verified pretraining cutoff, frozen sampling and delayed executable fills.

[FinCast](https://huggingface.co/Vincent05R/FinCast), [official code](https://github.com/vincent05r/FinCast-fts/tree/488b19d1d85fa2b3d4b93469530cefdcf1cc97a4) and [paper](https://arxiv.org/abs/2508.19609) support financial point/quantile forecasting. Forecasting capability is useful for a research benchmark; these inspected materials do not establish a verified after-cost NYSE trading edge. No training-free forecast should be interpreted as a trained profitable policy.

[LearnFinance SAC](https://huggingface.co/hajirazin/learnfinance-models-sac), inspected November 16, lists a training window through October 2, 2026, and evaluation CAGR about 56.5%, Sharpe about 1.74 and drawdown about 18.4%. The model card does not supply sufficient evaluation dates, benchmark, cost contract or reproducible fill ledger to validate those numbers. Earlier cached card values differ; this record uses the live inspected card. Status: claimed, not reproduced.

## 5. Proposed first experiment — not implemented or evaluated

Hypothesis ID: nyse_momentum_retention_buffer_v1.

Question: does retaining a still-highly-ranked stock improve net results over replacing every stock that leaves the top five, when predictions and all other assumptions are fixed?

Proposed controls:
- Point-in-time NYSE common stocks, including delisted names and corporate actions; no crypto, ETFs, shorts or leverage.
- Monthly ranking from completed observations using one 12-minus-1-month momentum rule. Proposed formula: close[t-21] / close[t-252] - 1, with an explicit corporate-action convention.
- Proposed eligibility: prior-session price above $5, trailing median dollar volume at least $20M, adequate valid history and no invented observations.
- Compare naive top-five selection with top-five / retain-through-top-ten selection. Rebalance monthly; fill vacancies deterministically by score and permanent security ID. No threshold sweep.
- Respect existing $5,000 research cash, $1,000 position cap, whole shares, settled-cash constraints and risk limits. Residual cash remains cash.
- Signals use information available before the hypothetical next-session fill. Use gap-aware execution and actual available quote data where required; do not silently present adjusted-close fills as executable opens.
- Charge all purchases/sales, exits, entry capital, exposure changes and drift-induced trades consistently. Preserve 43 bps per side stress and distinguish that stress from any independently measured execution cost.
- Freeze calendar, data versions, baseline and eligibility before evaluation. Historical diagnosis is separate from an unseen forward checkpoint. Correlated stocks on one rebalance date are not five independent observations.
- Report net expectancy, net return, drawdown, Sharpe, profit factor, turnover, trade/event-cluster counts, rejected fills, data gaps, costs and failures. Apply existing minimum-sample and confirmation gates; do not lower them to make a result pass.

Current blocker: no verified point-in-time NYSE security master plus matching versioned total-return/quote data was supplied or acquired in this audit. The three-stock daily replay in PR #290 cannot validate a broad stock-ranking claim. Reading code or running synthetic tests does not satisfy this data requirement.

## 6. Daily learning and continuity

A separate enabled automation, NYSE Edge Research, is configured for every day around 18:00 America/Los_Angeles, starting November 16. It searches sources, audits or reproduces one bounded candidate when feasible, and retains dated positive/negative evidence on a research branch for draft-PR review. It does not change GitHub scheduled workflows or resume the paused crypto collector.

Each record must distinguish discovered, source-audited, independently reproduced and unseen-confirmed evidence. Preserve exact source revisions, real input fingerprints, costs, cutoffs, trial count, rejection reasons and next action. Do not turn prose confidence into a trading signal. Do not repeatedly consume the same holdout or count reruns as new evidence.

Next action: specify and source the data for the single momentum/retention-buffer comparison, then implement a research-only experiment if those inputs pass availability checks. Keep paper/live order APIs disconnected. No neural weights have been trained by this audit.

Verification of this record: inspected actual source files and model cards; checked quoted figures against source tables; no executable strategy or neural-training validation was run. Documentation only.


## Evening continuation — 2026-11-16

[Source/data feasibility audit](../strategy_sources/2026-11-16-ranking-data-feasibility.md) and [machine-readable integrity results](2026-11-16-nyse-listing-snapshot-audit.json) extend this record without changing the proposed experiment.

New evidence: Microsoft's MIT-licensed Qlib replacement cap and minimum holding duration are different from rank-based retention; preserve a single naive-versus-buffer comparison. A Hugging Face price candidate requires subscription access and has explicit-symbol survivor coverage limitations. Its newer universe-loader claims were not reproduced.

Two original NYSE listing snapshots were independently parsed: 3,170 and 2,717 unique symbols; 2,094 in common. Both raw Git blob hashes matched repository metadata and four integrity assertion groups passed. The snapshots omit permanent IDs, explicit instrument types and row availability times; names include non-common instruments. Symbol-string changes do not establish delistings. Two snapshots do not establish complete historical coverage.

Profitability status unchanged: zero investment trials; no returns/costs measured and no model trained. Next action is a fixed-interval identity/type and delisting-price coverage audit before implementing or evaluating the momentum/buffer comparison. Both earlier draft PRs had no owner comments when checked. Keep PR #292 draft; no merge or runtime/configuration change.


## Identity continuation — 2026-11-17

[Historical issuer audit](../strategy_sources/2026-11-17-historical-issuer-identity-audit.md) and [JSON evidence](2026-11-17-nyse-issuer-join-audit.json) extend the next-action data work.

One actual historical exact-ticker join matched 2,708 of 3,170 NYSE listing strings to CIKs using a mapping published before the listing snapshot. Of those matches, 2,699 report NYSE and nine have an empty exchange. The remaining 462 listing strings stay unresolved. Seven integrity assertion groups passed, including both raw Git blob identities and count reconciliations.

There are 60 shared-CIK groups covering 150 matched listing rows; CIK identifies a filer rather than each individual security. Do not deduplicate share-class histories on CIK alone or discard unmatched symbols. The archived mapper's 2025 head is not a current-2026 security master. No security type, delisting-price coverage or executable quotes were established.

Investment trials remain zero. No returns, drawdown, Sharpe, profit factor or trade costs were measured; market-data investment evaluation was skipped and no model was trained. Next bounded action: audit historical security-registration tables, available before the same cutoff, for the three demonstrated shared-CIK pairs; then assess fixed-interval price/corporate-action/delisting coverage before evaluating the unchanged naive-top-five versus retain-through-top-ten proposal. PRs #290/#292 remain drafts with no owner comments observed; the crypto pause and runtime/configuration boundaries remain intact.
