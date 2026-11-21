# Twelve-stock market-data manifest audit — November 18, 2026

**Audit completed; dataset rejected for strategy evaluation.** This is a source/data audit, not a backtest, training run or executable-profit result. [Versioned manifest](../learning/manifests/2026-11-18-nyse-market-data-audit.json) records the frozen scope, calendar, exact query template, source revisions, raw-response SHA-256 fingerprints, SEC class evidence, exceptions and remediation requirements. `runner_ready` is false and `runner_manifest` is null: required fields were not invented to make the runner accept data.

## Scope frozen before outcome evaluation

CRM, NOW, SQ, SNOW, UBER, PINS, CHWY, TWLO, PLTR, NET, FSLY and ASAN. All twelve were retained. The deliberately narrow, retrospectively named software/consumer-platform scope supports only a scoped diagnostic; it is not a survivorship-neutral NYSE universe. No ranks, strategy returns or portfolio outcomes were evaluated before or after this audit.

History: December 1, 2020–February 2, 2022. Proposed development block: January 3–31, 2022. The monthly 12-minus-1 top-five versus top-five/retain-through-rank-ten specification remains unchanged, with $5,000 cash, whole shares, a $1,000 position entry cap and 43 bps per side. No additional variants or investment trials.

## Sources and evidence status

- Historical OHLCV, symbol, dividend and split tables: [post-no-preference/stocks](https://www.dolthub.com/repositories/post-no-preference/stocks), pinned Dolt revision `uuk0i93qvs5rrk9k8geq250699j4l71c`. The API log reports `2022-02-03 10:00:05.161`, without an offset. The manifest retains the exact endpoint/query template and metadata queries. Broad year-long queries timed out; bounded date queries succeeded. No data was purchased.
- Primary [NYSE calendar published December 9, 2019](https://ir.theice.com/press/news-details/2019/NYSE-Group-Announces-2020-2021-and-2022-Holiday-and-Early-Closings-Calendar/default.aspx). All applicable closures and December 24, 2020 / November 26, 2021 early closes were included, with America/New_York daylight-saving offsets.
- SEC 10-Q covers filed before December 31, 2021 establish an NYSE common-stock class for each of twelve candidates. The manifest records each accession, title, issuer CIK, catalog-reported acceptance timestamp, immutable filing URL and byte hash. Current SEC submission catalogs were discovery aids, not historical security masters. Each saved original filing was re-parsed using the existing offline cover reader. CIK is not a permanent class ID.
- Prior [January coverage audit](2026-11-17-january-2022-coverage.md) had already found a genuine DDS dividend omission in this provider. DDS is outside this fixed scope; its defect is a reason to require independent completeness checks, not evidence that these twelve had a dividend.

## Measured data checks

| Check | Result |
| --- | ---: |
| Weekday queries, including holiday controls | 307 / 307 succeeded |
| Expected NYSE sessions | 296 |
| Holiday controls returning no rows | 11 |
| Expected symbol-session cells | 3,552 |
| Observed unique rows | 3,504 (98.65%) |
| Missing cells | 48 |
| Present rows passing basic OHLCV constraints | 3,497 |
| Invalid OHLC rows | 7 |
| Duplicates / unexpected symbol-date rows | 0 / 0 |
| Pre-block calendar sessions | 274 |
| January evaluation sessions | 20 |
| Following settlement sessions | February 1 and 2, 2022 |
| SEC common-stock cover matches | 12 / 12 |
| Provider dividend / split rows in scope | 0 / 0; completeness unknown |

Every stock lacks May 3, May 14, July 7 and July 8, 2021. Four separately issued `COUNT(*)` queries without a symbol filter returned zero rows for the entire provider OHLCV table on those dates at the same revision. These are source coverage gaps, not exchange holidays or grounds to shorten the lookback calendar. No interpolation, forward fill or deletion of securities was performed.

FSLY, NOW, PINS, PLTR, SNOW, TWLO and UBER each have an opening price above the daily high on May 5, 2021. The exact seven source rows are retained in the manifest. For example, FSLY open 61.33 exceeds high 60.24. The cause and correct prices were not established; they were not silently repaired. Basic consistency passing for other rows does not verify vendor adjustment conventions or venue execution availability.

## Readiness decision and exact remaining work

1. **Source integrity fails.** Reconcile the four missing sessions and seven inconsistent bars against independently sourced, documented data. Preserve the frozen universe, calendar and negative findings.
2. **Historical availability is unproved.** This February 3 snapshot is later than every January decision. It has no per-row first-publication times. Do not backdate current acquisition times or assume bar-close availability. A historical diagnostic needs admissible earlier vintages with verified timestamp semantics. A separately frozen prospective collection could instead record actual receipt times without claiming historical availability.
3. **Corporate-action completeness and price basis are unproved.** Empty provider tables do not certify no events. Reconcile all twelve classes over the entire warmup/evaluation interval against primary action records, including renames, conversions, exit consideration and payment dates. Verify raw OHLC conventions and construct the causal total-return momentum index. ATR14 is not calculated from a defective/unverified price history.
4. **Class continuity remains incomplete.** Covers resolve type at filing dates; assign documented permanent class-level IDs and verify eligibility intervals and exits. Do not substitute CIKs or tickers for security identities.
5. **Execution evidence remains absent.** A complete daily-bar diagnostic would still lack order-time bids, asks, size, halts and auction eligibility. After valid signals freeze orders, inspect following-session execution evidence. The 86-bps stress scenario is not a measured execution-cost estimate.
6. **Rights and expenses are not established.** Public data access does not establish upstream redistribution rights. Raw files remain outside Git; source URLs/hashes are retained. Unknown data, hosting and brokerage costs remain unknown, not zero.

Immediate next bounded action: reconcile the four missing dates and May 5 inconsistencies against one independently documented historical OHLC source, without changing this scope. If historical publication and adjustment provenance cannot be established, reject this bundle permanently for causal evaluation and prepare a separately frozen prospective diagnostic with observed receipts. Filling price holes alone does not clear the remaining gates.

## Verification and research accounting

The [offline auditor](../learning/tools/audit_nyse_manifest.py) was run on downloaded byte-exact source responses and twelve original SEC filings. It checked file lengths/SHA-256, re-parsed filing covers, built the primary-source calendar, inspected all 3,504 price rows, reconciled missing/invalid cells and verified the four whole-table controls. A second run produced a byte-identical manifest. Count/class/status reconciliation assertions passed. These are acquisition/audit checks, not strategy tests. The acquisition directory is transient; the manifest preserves exact public source locators and hashes, not a claim that all raw files have been durably distributed.

```sh
python research/learning/tools/audit_nyse_manifest.py RAW_ACQUISITION_DIR OUTPUT_JSON
```

The raw acquisition directory contains `freeze.json`, `daily-source-audit.json`, `daily/YYYY-MM-DD.json`, `security-audit.json`, its SEC catalog/filing sources, `sources.json`, `dividends.json`, `splits.json`, `symbols.json`, `version.json`, `missing-session-controls.json` and its four responses. Source indexes are generated by acquisition; the published manifest is an audit index, not a drop-in runner input. Raw-response fingerprints can detect changed provider payloads during re-acquisition.

One bounded data audit; **zero strategy investment trials, training runs or unseen confirmations**. Market evaluation skipped. Net return, drawdown, Sharpe, profit factor, trade counts, turnover, stress execution totals and expected profit remain unmeasured/null. No evidence of a profitable strategy was gained or lost because no strategy was evaluated. This source revision lost evidence of suitability as a complete signal-history input. The twelve instrument-type decisions gained direct primary-source support.

No workflow, secret, cloud/deployment, brokerage, trading gate or model tag changed. Keep PR #292 draft; no merge.
