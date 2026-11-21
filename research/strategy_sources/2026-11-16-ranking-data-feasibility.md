# NYSE ranking feasibility audit — 2026-11-16 evening

## Decision

Keep the proposed momentum/retention-buffer comparison as the first research priority. Free historical listing snapshots partially address membership reconstruction, but they do not yet supply an investable common-stock universe or matching delisting-aware execution data. No profit claim advances.

Evidence states: Microsoft Qlib turnover control = source-audited engineering reference; GitHub NYSE snapshots = two actual files independently parsed and integrity-checked; Hugging Face price candidate = card-audited, gated and not downloaded. Market-data investment evaluation: **SKIPPED**. Model training: **NOT RUN**. Investment trials: **0**. Net return, drawdown, Sharpe, profit factor, trades and execution costs: **not measured**, not zero.

## Continuity and scope

Read main's tree and .github/copilot-instructions.md, draft PRs #290/#292, their empty comment timelines, the November 14 source audit and the latest November 16 learning record. No AGENTS.md was listed in either inspected complete tree. The paused collector still exposes only manual dispatch. PRs remain drafts; no merge or production change is made.

## New engineering source: Microsoft Qlib

Pinned [source revision](
https://github.com/microsoft/qlib/tree/54355232463878d2eebb91fe0ee5fa7fa1f5976c) is 54355232463878d2eebb91fe0ee5fa7fa1f5976c. Inspected qlib/contrib/strategy/signal_strategy.py, the Alpha158 LightGBM example configuration and the MIT license.

The [TopkDropoutStrategy implementation](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/strategy/signal_strategy.py) limits replacements using n_drop and enforces a minimum holding duration. It obtains scores from the preceding calendar step. This differs from retaining a stock until its rank exceeds ten: a replacement cap, a holding-duration restriction and rank hysteresis are three distinct interventions. Do not combine them in the first comparison.

The strategy estimates buy allocations from temporary cash after simulated sales. Its explicit purchase-fee budget adjustment is commented out; downstream exchange behavior was not audited here, so this is not a demonstrated overspending defect. A separate settled-cash and fee-inclusive quantity check is required for our $5,000 account.

The [example configuration](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml) uses CSI300, 50 holdings, 100 million capital units, close-price fills and different costs. Training ends in 2014, validation spans 2015–2016 and tests span 2017–August 2020. Its settings do not reproduce our NYSE/whole-share/next-session/86-bps contract. No Qlib or model code was downloaded or executed.

## Hugging Face data screen

The [original Stocks-Daily-Price card](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price/blob/main/README.md), inspected November 16, identifies subscription-gated downloads and acknowledges substantial survivor selection in an explicit-symbol load. It also advertises a newer historical liquid-universe loader, which was not downloaded or independently validated. Neither layer is established as NYSE common-stock history with permanent identities and actual next-session quotes. Do not conflate a newer universe feature with repaired price coverage.

The loader normally adjusts OHLC using an adjustment ratio. Adjusted historical opens must not silently become executable prices or whole-share purchase prices. No access request, subscription, account or purchase was made. Status: not an available free input for this experiment.

A [Hugging Face mirror card](https://huggingface.co/datasets/yolo22/stock-pit-archives) pointed to the original rreichel3/US-Stock-Symbols repository. The mirror's quality claims were not accepted as validation; the original files were checked below. Its two bundles were not downloaded.

## Independent two-snapshot listing check

Fetched only pinned JSON data from [the original listing repository](https://github.com/rreichel3/US-Stock-Symbols). No foreign code ran. Source selection cutoff was 2026-11-17 02:59:46 UTC; commit timestamps below are repository publication timestamps, not exchange event times.

| Property | Older snapshot | Newer snapshot |
| --- | --- | --- |
| Commit | b02d9d7ace317f861f8a6f59e2cfa88e207f3ce9 | 4cf8c5b8ffa1da595ac4da28645a1b5d58d5e658 |
| Commit time, UTC | 2022-01-01 00:11:00 | 2026-11-17 00:23:27 |
| JSON rows / unique nonempty symbol strings | 3,170 / 3,170 | 2,717 / 2,717 |
| Empty / duplicate symbol rows | 0 / 0 | 0 / 0 |
| Explicit security-type field | absent | absent |
| Permanent ID field | absent | absent |
| Row event/receipt/as-of timestamp | absent | absent |

Intersection: 2,094 symbol strings. Older-only: 1,076; newer-only: 623. These differences are **not confirmed delistings**: transfers, renames, instrument changes and source coverage remain unresolved.

Names containing the exploratory keywords preferred, depositary, warrant, ' etf' or ' unit' numbered 586 and 525. These are heuristic hits, not security classifications; depositary interests and units need explicit policy. Direct sample inspection showed preferred instruments and warrants among listings. Do not derive common-stock eligibility from names alone.

The [machine-readable audit](../learning/2026-11-16-nyse-listing-snapshot-audit.json) retains byte counts, raw SHA-256, Git blob IDs, schema and exact URLs. Raw Git blob IDs matched connector metadata for both files. Four assertion groups passed: byte identity, reconciled set counts, unique/nonempty symbols, and missing critical schema fields. These are data-integrity checks, not trading tests. Only two snapshots were inspected; full-history coverage and scrape fidelity remain unverified. GitHub metadata returned no repository license; data-reuse rights remain unverified. Raw listing rows are not redistributed in this PR.

## Next executable action

Resolve persistent security identities and historical common-stock classifications for one fixed diagnostic interval, then audit daily price, corporate-action and delisting coverage against those eligible IDs. A historical snapshot may enter an as-of join only after its recorded publication; do not backdate it to a claimed scrape day. Missing snapshots must produce an explicit staleness policy rather than future filling.

Before evaluating returns, freeze the calendar and data hashes for the existing single momentum/buffer proposal. Keep naive top-five and retain-through-top-ten as the only arms; do not add Qlib dropout or minimum-hold variants after seeing results. Charge actual buys and sells at 43 bps per side, budget fees before whole-share purchases, preserve settled cash and the $1,000 cap, and reject unavailable inputs rather than excluding names because their future returns are missing.

No broker, order, active model, risk, promotion, deployment, cloud, secret or workflow change. Sources gained feasibility/implementation evidence only; none gained profitability evidence.
