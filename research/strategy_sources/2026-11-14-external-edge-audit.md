# External strategy edge audit — 2026-11-14

No source reviewed establishes a definite executable edge for this bot's NYSE common-stock, cash-only, 86-bps stress-cost contract. This is a source and code audit, not a reproduction of the authors' returns. One strategy family merits a controlled test: opening-range breakouts conditioned on unusual opening volume.

## Ranked findings

| Source | Evidence found | Decision for this bot |
| --- | --- | --- |
| Stocks-in-play opening-range breakout (ORB), author paper, 2024-02-16 | Opening-activity selection improves reported results. | **Maybe — first research priority.** Published shorts, leverage and cost assumptions differ from our contract. |
| `hsaeed22058/strategy-backtest`, GitHub, inspected 2026-11-14 | Public implementation and unusually candid bias disclosures; direct engine inspection found unresolved fill and timing problems. | **No as delivered.** Useful reference, not trusted execution logic. |
| `liu-xuhui/us-stock`, GitHub, April 2026 | Chronological stock-ranking research, but its own results turn negative at costs below our stress hurdle. | **No under the existing cost contract.** Retain data-engineering lessons. |
| Kronos, Hugging Face / arXiv, 2025-08-02 | A financial forecasting model with published temporal separation and an investment simulation. | **Maybe as a later benchmark; no demonstrated NYSE edge.** |
| Part Time Larry ORB material, YouTube / author site, July–August 2025 | Identifiable ORB implementation tutorials and author material. | **Engineering reference only.** Full videos were not accessible during this audit. |
| ORB discussion, Reddit r/algotrading | A public code discussion linked to the same strategy family. | **Discovery source only.** Repeated discussion of one paper is not independent replication. |
| `TauricResearch/TradingAgents`, GitHub | Multi-agent analysis tooling; its own reproducibility notes identify moving social/news inputs. | **No as historical edge evidence.** |
| `kylegothman/trading-system`, GitHub | Architecture/validation showcase explicitly withholding strategy code. | **No reproducible edge established.** |

“Maybe” authorizes only research prioritization. It is not a feature value, inference label, trading gate or recommendation to allocate capital.

## 1. Best lead: opening-range breakout plus relative volume

[Zarattini, Barbon and Aziz, A Profitable Day Trading Strategy For The U.S. Equity Market](https://concretumgroup.com/wp-content/uploads/2026/02/A-Profitable-Day-Trading-Strategy-For-The-U.S.-Equity-Market.pdf), first version 2024-02-16.

Reported Sharpe rises from 0.48 for basic ORB to 2.81 with opening-activity selection. The long/short experiment permits 4x leverage and charges $0.0035/share commission; it does not validate our cash-only, 86-bps adaptation.

### Public engine audit

Inspected repository revision: [`f90d5f55d512c08c88bff43253a3170d2009ce12`](https://github.com/hsaeed22058/strategy-backtest/tree/f90d5f55d512c08c88bff43253a3170d2009ce12), committed 2026-07-23.

[Engine permalink](https://github.com/hsaeed22058/strategy-backtest/blob/f90d5f55d512c08c88bff43253a3170d2009ce12/stratbt/engine.py), blob SHA `271b53e355885f1bed5e46d3b003390c8951970c`.

Direct code observations:

- Exit evaluation begins at `entry_pos + 1`. A stop crossed during the entry bar is ignored, including when the breakout order filled at that bar's open and the subsequent low necessarily crossed the stop.
- The optional time-stop uses the current bar's high/low to calculate favorable excursion, then can fill at that same bar's open. That decision uses information arriving after the assumed fill.
- The optional VWAP condition reads the completed entry bar, although the breakout fill can occur before that bar ends.
- Stop fills subtract a fixed fraction of initial risk from the stop level; the function does not generally price a gap at the first available executable quote.

These are code-audit findings, not a claim that we reproduced or quantified their profit impact. No foreign trading code was installed or executed.

[Repository disclosures](https://github.com/hsaeed22058/strategy-backtest/blob/f90d5f55d512c08c88bff43253a3170d2009ce12/README.md) also describe survivor coverage limitations, IEX-only intraday data, hindsight in variant construction and fixed-base performance denominators. Walk-forward selection does not repair missing delisted names or an invalid fill engine.

### Single proposed NYSE test

This is a proposed adaptation, **not yet implemented or a validated replication**. Freeze its implementation and calendar before examining new test outcomes; do not search alternate thresholds, assets, exits or horizons.

- Eligible: point-in-time NYSE common stocks, including delistings; opening price >$5; prior-14-session mean volume ≥1 million shares and ATR >$0.50.
- Measure the completed 09:30–09:35 America/New_York range and volume. Relative volume equals that volume divided by the mean volume in the identical interval over the previous 14 complete sessions. Reject missing intervals.
- Rank eligible names once by relative volume, require at least 1.0, retain the top 20, and resolve ties by permanent security ID. Within that fixed set, consider only positive opening ranges; do not replace bearish or untradeable names with hindsight.
- Observe the first later completed one-minute close above the range high. A hypothetical entry cannot precede input availability plus latency; use the first minute boundary strictly after that time and an executable ask, with gap treatment. This close-confirmed, delayed entry differs from the paper.
- Limit to one long entry per stock/session. Proposed stop: entry minus 0.1 prior ATR; include the entry interval and adverse gaps. Exit before the scheduled session close using a decision and quote available before the fill.
- Preserve the existing cash, whole-share, position, risk and drawdown restrictions. Model settled cash, spread, participation, missing quotes, halts and unfilled orders. Apply 43 bps each side under the shared stress contract.
- Primary question: does the frozen adaptation add positive net expectancy beyond the frozen baseline? Count correlated stock trades on one date as one event cluster for sample-size checks. Require at least 20 independent observations in every predeclared block and genuinely unseen forward confirmation. Published historical samples are diagnostic, not untouched confirmation.

Required data: exchange/security history, delistings and corporate actions; consolidated one-minute trades/volume including opening auctions; contemporaneous NBBO/size and halts; session calendar; source event timestamps and actual receive/availability timestamps. Revised historical bars must retain retrieval/version metadata. Historical availability times cannot be invented from bar-end timestamps.

The existing three-stock daily-bar replay cannot establish this cross-sectional intraday hypothesis. Missing complete opening-volume coverage or executable quote data blocks a meaningful test. The expected direction is conditional upside continuation after a positive range breakout; volatility, opening-auction composition, market-wide gaps and crowding are confounders.

## 2. Stock-ranking ML: rejects at the current cost hurdle

[Xuhui Liu, Daily U.S. Equity Alpha Research](https://github.com/liu-xuhui/us-stock/blob/d74205a7945f80b97980cce163e756a92edbd981/report.md), April 2026; revision committed 2026-04-29.

The author reports 2024–2025 long-only cumulative return of 41.66% before costs, 16.36% at 10 bps per side and **−4.42% at 20 bps per side**. The reported negative 40-bps round-trip result already fails below our 86-bps hurdle. These are author results, not our rerun; no exact 86-bps result is claimed.

The [backtest code](https://github.com/liu-xuhui/us-stock/blob/d74205a7945f80b97980cce163e756a92edbd981/src/backtest/run_backtest.py) charges turnover costs but books five-day sleeve PnL on exit rather than marking every held position daily. It also filters on future-row availability, requiring an as-of/universe audit. Licensed raw data are absent. A replacement with lower turnover would be a new hypothesis, not a validated repair.

## 3. Hugging Face: model competence is not executable profit

[Kronos model card](https://huggingface.co/NeoQuasar/Kronos-base) and [paper v1](https://arxiv.org/html/2508.02739v1), 2025-08-02.

The paper separates training through June 2024 from tests beginning July 2024. Its investment simulation, however, uses Chinese CSI 300/800 constituents and 0.15% per-trade costs. It does not establish after-cost NYSE execution under our contract. A later benchmark would need a pinned model revision, frozen sampling/seeds, actual as-of inputs and delayed executable fills. No weights were installed and no neural engine was retrained.

[ParallelLLC/algorithmic_trading](https://huggingface.co/ParallelLLC/algorithmic_trading) is another useful HF discovery: validation/backtesting and FinRL tooling. Its model card describes research infrastructure, not independently verified NYSE strategy profitability. Validation features in a README do not establish that any supplied strategy passes them.

## 4. Social/video and other GitHub evidence

[Part Time Larry, How to Code an Opening Range Breakout with IBKR](https://www.youtube.com/watch?v=a1813iCcsWQ), 2025-08-09, and [author's Backtesting.py ORB article](https://hackingthemarkets.com/backtesting-py-tutorial/), 2025-07-16: identifiable implementation resources. YouTube fetching was throttled and the article body is subscriber-only. Searchable metadata and listings were inspected; this audit does not claim to have watched the full videos, read the paid article, or verified broker statements.

[Reddit ORB code discussion](https://www.reddit.com/r/algotrading/comments/1hezqql/opening_range_breakout_for_stocks_in_play_code/): useful discovery trail, but repetition of the same research is not additional independent evidence.

[TradingAgents](https://github.com/TauricResearch/TradingAgents): its README explicitly says live news, StockTwits and Reddit can reflect the present even for a historical trade date. That permits future information in a historical analysis unless inputs are separately archived and restricted by availability time. Report quality or agent debate is not a measured return series.

[kylegothman/trading-system](https://github.com/kylegothman/trading-system): the showcase says strategy code is not included. Its public claims therefore cannot be independently reproduced from this repository; “live since” is not a publicly verified fill ledger.

## Implication for this bot

The earlier negative replay tested a particular daily SMA rule on three stocks. It does not establish that every strategy, every stock-selection method or the neural system lacks an edge. Equally, adding books, a forecasting model or more indicators does not establish profitability.

The actionable gap is now explicit: point-in-time session selection and intraday execution data for one frozen ORB hypothesis. Keep source discovery separate from measured market features, and retain rejected sources so a future search does not repackage the same claims as new evidence.

This audit changes research priorities only. It does not change active inference tags, risk limits, trade gates, leverage or orders.
