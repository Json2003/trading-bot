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

## Six security classes resolved on November 17, 2026

The three issuer pairs already selected in the previous entry yielded six historical class decisions. UNM, BDX and DDS pass the common-stock type screen. UNMA was debt, BDXB represented preferred stock, and DDT was issued by Dillard’s Capital Trust I. The separate trust issuer, its prospectus and the historical ownership filing explain why DDT does not appear on the parent’s quarterly cover. The prior join’s 462 unmatched symbols still require resolution. [Source audit](../strategy_sources/2026-11-17-sec-security-class-audit.md)

The offline filing reader extracted 2, 11 and 1 title/symbol/exchange rows from the three 2021 covers. Five target tickers matched directly. DDT required separate records. A deliberately mismatched context was rejected and all three command-line outputs reproduced their recorded semantic hashes. The issuer report and OCC memo also establish a real accounting test for DDS’s $15 special dividend, ex-November 26 and payable December 15, 2021. [Evidence and fingerprints](2026-11-17-sec-class-evidence.json) · [Offline extractor](tools/sec_cover_rows.py)

One class audit, zero investment trials, zero model-training runs and zero unseen confirmations. Market evaluation was skipped. Net return, drawdown, Sharpe, profit factor, trades, turnover and realized costs remain unmeasured. Next, check historical data coverage for January 3–31, 2022, with the starting universe and gaps retained. Verify class-specific prices, opening quotes, corporate actions and terminal proceeds before testing the proposed top-five and retention-buffer arms. The public Daily TAQ sample directory inspected does not include this January window.

## January 2022 execution-data check, November 17, 2026

One fixed-window audit covered 3,170 starting strings across all 20 opens, including the 462 still lacking issuer matches. There were 53,519 exact daily price rows in 63,400 possible symbol-days, with 53,376 passing basic price and volume checks. Two historical mergers and a ticker change appeared one open late in the listing source. The dated price-source revisions omitted Dillard’s verified $15 dividend and added 928 January dividend rows between February 1 and February 3. SL Green’s reverse-split terms matched the issuer when the source eventually added the row. [Source audit](../strategy_sources/2026-11-17-january-2022-coverage.md) · [Per-session evidence and fingerprints](2026-11-17-january-2022-coverage.json)

Reject this source combination for a standalone executable backtest. The Nasdaq BX monthly execution-quality statistics and the pinned Hugging Face minute-bar archive yielded no January per-order opening bids, asks and sizes. The historical bar archive gains evidence for coverage diagnostics and loses support as a point-in-time total-return source. Investment trials, model training and unseen confirmations were zero. Performance and realized costs remain unmeasured. Next, reconcile the January 31 development ranking’s required 252-session history and known corporate events. Freeze both arms’ intended orders before checking the relevant next-session quotes. Keep all unresolved names and unfilled orders visible.

## Operating structure proposed on November 18, 2026

The [operating plan](2026-11-18-nyse-operating-structure.md) maps the existing NYSE framework and data audits to one proposed research entry point, a shared portfolio ledger, frozen strategy comparison, validation and owner report. Direct runner inspection confirms immediate reuse of sale proceeds, dividend accrual without payment dates and rejection of split-bearing histories. The implementation priority is a manifest-fed offline replay with settled-cash and corporate-action regression cases, then the existing top-five versus retain-through-rank-ten comparison.

The 3,170-string coverage study does not require completing a market-wide security master before coding and software tests. A smaller predeclared universe can support a scoped diagnostic with its selection and limitations recorded. Executable return claims still need defensible order-time fills and a complete applicable cash ledger. Existing risk controls and independent-observation gates apply. This entry is a proposed operating specification. Zero new investment trials, training runs or market evaluations were completed.

## Offline runner implemented on November 18, 2026

The implemented [offline command and data contract](nyse-offline-runner.md) now connect a versioned local manifest, the frozen monthly top-five/retention comparison, a shared Decimal cash ledger, validation metrics and an owner report. Reused components include the research safety flags, NYSE nonoverlap counting convention and risk parameters, with an explicit adapter to the governed result schema. The standalone CLI keeps acquisition outside the offline run.

Cash accounting now distinguishes settled cash, sale receivables, dividend receivables and fractional entitlements. Purchases reserve stress fees and use whole shares. Corporate actions retain permanent security identity and cost basis. Intended quantities are planned at the prior decision, with gap violations rejected and cash-constrained vacancies retried at a later close. Monthly targets supply this experiment's holding rule. The legacy SMA diagnostic's twelve-session exit belongs to its own strategy. The preserved 0.5-percent entry sizing budget measures gross stop distance before fees.

All 26 new tests passed. They exercise settlement across the T+1 transition, the audited dividend/conversion terms, gap rejection, entry-bar stops, both portfolio arms, missing/late input failures, explicit units, actual CLI exit behavior and concurrent retries. Six existing research-contract functions also passed with their temporary-directory fixture supplied directly. The [verification record](2026-11-18-nyse-offline-verification.json) retains input and source hashes. All investment inputs in these execution checks were synthetic. Their performance figures are software examples.

Zero new investment trials, training runs or unseen confirmations. Genuine market-data evaluation remains skipped because the earlier data audit lacks the complete admissible signal history, reconciled actions and order-time quotes needed for an executable claim. The runner currently supports development diagnostics with explicitly labelled daily-bar fill assumptions. Next, prepare the first audited development manifest and run the frozen comparison. Preserve negative results and the original monthly observation IDs when reproducing a run.

## Market-data manifest audit completed on November 18, 2026

The [twelve-stock source audit](../strategy_sources/2026-11-18-nyse-market-manifest-audit.md) and [versioned audit manifest](manifests/2026-11-18-nyse-market-data-audit.json) complete one bounded, predeclared data check. All twelve pre-2022 SEC covers were re-parsed and identify NYSE common-stock classes. The 307 weekday queries succeeded; the primary calendar expects 296 sessions and 3,552 symbol-session cells. There were 3,504 unique rows, of which 3,497 pass basic OHLCV consistency. Four source-wide missing dates account for 48 absent cells; seven May 5, 2021 openings exceed their highs. No names were dropped or bars repaired. Two independent offline audit executions produced a byte-identical manifest.

The audit is complete and the dataset is blocked. The February 3, 2022 snapshot cannot establish availability at January decisions; raw/adjusted conventions, causal total returns, interval-wide action completeness and permanent class continuity are still unverified. The manifest explicitly sets runner_ready=false and runner_manifest=null rather than fabricating required fields. Order-time executable quotes and upstream redistribution rights remain absent/unverified.

One data-audit cycle; zero new investment trials, training runs or unseen confirmations. No rankings or strategy returns were evaluated. Profit, drawdown, Sharpe, profit factor, trades, turnover, execution costs and operating expenses remain unmeasured; no purchases made. The source loses support as a complete historical signal input; security-type evidence improves. Next, independently reconcile the four missing dates and seven inconsistent bars without changing the frozen scope, then clear publication/action/identity gates or reject the historical bundle and freeze a prospective diagnostic. Do not relabel this audit as a ready runner manifest or a completed market run. Preserve the crypto pause and PR #292 draft status.

## Independent-provider reconciliation continued on November 18, 2026

The [reconciliation source audit](../strategy_sources/2026-11-18-nyse-price-reconciliation.md) and [versioned evidence](manifests/2026-11-18-nyse-reconciliation.json) retain the same twelve names and 296-session calendar. The separately retrieved Yahoo/CSI history contains all 3,552 expected cells and coherent counterparts for all 48 missing cells and seven invalid bars. With the issuer-documented later NOW split undone for comparison and historical SQ mapped to the documented later XYZ query alias, all 3,504 overlapping opens and closes agree within one cent. There are eleven high disagreements and one low disagreement, all on May 5, 2021. Four high disagreements were missed by internal OHLC validity alone. Most exact volume differences are small, but 23 exceed 1% of the larger provider figure; vendor volume definitions remain unproved. Separate retrieval is established, full upstream independence is not.

Twelve original annual filings support retrospective no-cash-dividend findings. The 2021 Square/Block name announcement retained SQ/CUSIP and required no action by existing holders. The January 2022 Afterpay acquisition is not a receipt or conversion owed per already-held SQ share. Empty event lists and no split-keyword matches do not certify a complete class/action history. No complete-through certificate was created.

Reject the retrieved historical bundle for causal January 2022 strategy evaluation: the February 2022 snapshot and November 2026 retrieval do not establish original row availability. No fields were backdated, original bars were not replaced, and no strategy was run. Source hashes, code, decision metadata, failed lookups and a byte-identical offline replay are retained. Zero investment trials, training runs and unseen confirmations; profit/expenses remain unmeasured. The source reconciliation and dividend evidence advanced, but neither strategy gained executable-profit evidence. The next admissible input must have defensible historical vintages or belong to a separately frozen prospective diagnostic with observed receipts and complete applicable actions. Do not spend another cycle merely patching this historical bundle and calling its timing resolved. No collector, schedule, brokerage, cloud resource or trading gate changed; keep PR #292 draft.

## Working market-data input path on November 18, 2026

The runner now blocks an incomplete price window before either arm runs, including cash-only windows. A buy also checks that its security is still eligible at the following open. The [capture and preparation commands](nyse-forward-capture.md) enforce original receipt times, reproducible source pins, reviewed corporate actions and a complete observed window. The rejected January data remains preserved. Two earlier archive revisions still lacked the four sessions and contained the seven invalid May 5 bars.

The separate forward window starts December 1, 2026 and ends January 29, 2027. Latest SEC covers establish ten NYSE names among the twelve registered classes. Fastly and Palantir now list on Nasdaq. The initial real snapshot is complete for 262 sessions through November 17, with 3,144 raw rows and 2,620 eligible rows. The independent price check and four Salesforce dividends plus the ServiceNow split are recorded in the [source audit](../strategy_sources/2026-11-18-nyse-forward-data-repair.md). Restoration reproduced the original capture bytes and receipt times. The feed lacked November 18 during the evening check. The first decision requires an actual receipt by 6 a.m. Pacific on December 1. Collect those bars before that cutoff and close later action and membership reviews before running the frozen comparison.

The real-data preparation run is waiting for forward observations. Monthly strategy trials, training and unseen confirmation remain at zero. Expected profit and trading results remain unmeasured. The first two-month window does not satisfy the twenty-independent-observation requirement. The source audit and [verification record](2026-11-18-nyse-forward-verification.json) distinguish software validation from data readiness.

## Receipt-chain hardening on November 18, 2026

The provider's latest pinned revision still omitted all twelve requested November 18 bars during this run, so market evaluation remained blocked and no strategy trial was created. The source remained at revision `f5psmujj89khlc43g5kicsclpd0orhok`. This is a publication delay, not a zero-return session.

One recoverable implementation gap was fixed before any replay manifest can be admitted. A generated market manifest now includes the SHA-256 of its frozen capture plan, the SHA-256 of the reviewed membership/action packet and every raw capture receipt's revision, byte hash, date interval and partition receipt metadata. The offline loader rejects a market manifest that omits or malforms this chain. Previously, those facts appeared in the preparation report but were not part of the immutable manifest identity used by a run.

All 32 focused NYSE software tests passed after the change, including generation and loading of a complete fixture plus rejection of the same manifest with its source evidence removed. Synthetic fixtures remain software checks only. Market strategy trials, independent observations, model-training runs and unseen confirmations remain zero. Profit, costs, return, drawdown, Sharpe, profit factor, turnover and fills remain unmeasured. Expected profit is unestimated.

Next executable action: capture the first newly published November 18 session without backdating its receipt, continue through the November 30 signal close, extend membership and corporate-action reviews through the evaluated window, then run the input-only check before either frozen strategy arm.

## November 18 bars recovered on November 19, 2026

The new pinned revision supplied 12 November 18 bars. All 48 OHLC comparisons agree within $0.0101 against the separately retrieved history. The scoped review of listings and corporate actions now reaches November 18 and preparation validated 2,630 eligible signal inputs. Both price packets reproduced byte for byte. The latest closed NYSE session was November 19. Its bars were still pending in the pinned source at the afternoon check. [Source audit](../strategy_sources/2026-11-19-nyse-forward-session-reconciliation.md) · [Verification record](2026-11-19-nyse-forward-verification.json)

The daily research task moves to 5 a.m. Pacific beginning November 20. The first decision cutoff remains December 1 at 6 a.m. Pacific. Seven more session captures are needed through the November 30 signal close. Market evaluation skipped, with zero investment trials and zero independent investment observations. The new bars improve data coverage. Expected profit remains unestimated.
