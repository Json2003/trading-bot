# Historical NYSE issuer-identity audit — 2026-11-17

## Outcome

An exact historical ticker-to-CIK join supplies partial issuer coverage, but is rejected as a complete security-identity solution for the proposed NYSE momentum/retention-buffer experiment. This is new data-feasibility evidence, not a backtest or a profitable strategy.

One data-join diagnostic was run. Investment trials: **0**; market-data investment evaluation: **SKIPPED — incomplete security classification/identity and delisting-aware prices**; model training: **0**. Net return, drawdown, Sharpe, profit factor, trades and execution costs were not measured. The $5,000 cash, whole-share, $1,000 position and 86-bps round-trip stress contract is unchanged.

## Continuity

Read current main's complete tree and repository instructions, PR #290 and its prior source audit, and PR #292's latest journal, source audit and integrity artifact. Both PR comment timelines were empty. No AGENTS.md appeared in the inspected complete trees. PR #290 remains draft and unmerged. Main's collector still has manual dispatch and no schedule.

The preceding audit found no permanent security IDs or explicit security types in NYSE listing snapshots. This study follows that specific gap rather than rerunning investment results or adding another model.

## Sources and prospective availability discipline

The [Hugging Face archive card](https://huggingface.co/datasets/yolo22/stock-pit-archives) provided the second original source, jadchaar/sec-cik-mapper. Its mirror bundles were not downloaded. The [original GitHub repository](https://github.com/jadchaar/sec-cik-mapper) is archived; the head inspected was 7883b83389836f9bba9bdfe53031467235746334, dated February 28, 2025. Its MIT software license was inspected, but current-2026 membership must not be inferred from that frozen head.

Instead, the join used these immutable original data files:

| Input | Commit | Repository publication, UTC |
| --- | --- | --- |
| NYSE listing JSON | b02d9d7ace317f861f8a6f59e2cfa88e207f3ce9 | 2022-01-01 00:11:00 |
| SEC-derived mapping CSV | b74d1efd3e005df08da0a38a4eedfcbdf2a02283 | 2021-12-31 00:03:08 |

The mapping predates the listing by approximately 24.13 hours. These commit timestamps are repository publication evidence, not exchange event times or independently verified source receipt timestamps. They cannot support fills earlier than publication, nor continuous full-history coverage.

The SEC [defines CIK at the filer/entity level](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/look-central-index-key-cik-number) and [does not guarantee the accuracy or scope of ticker associations](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). A filer key is not automatically an individual security key. The original mapper CSV exposes only CIK, Ticker, Name and Exchange; it supplies neither common/preferred classifications nor security-level validity intervals. Source mapper code was inspected, not imported or executed.

## Frozen diagnostic policy and actual checks

Before reading join counts, fixed exact symbol-string equality only. No fuzzy company-name matching, punctuation rewriting or new eligibility rule was selected after results. Ambiguous and missing matches stay unresolved. The universe was not restricted using future price or return availability.

A first-party Python standard-library audit downloaded only the pinned CSV and JSON data. It verified Git blob identities against connector metadata, parsed CIK as text to retain leading zeros, required ten-digit CIKs and unique mapping rows for accepted matches, and counted exchange agreements and shared issuer keys. No third-party trading or model code ran.

| Measured diagnostic | Result |
| --- | ---: |
| NYSE listing rows / unique symbol strings | 3,170 / 3,170 |
| SEC-derived mapping rows, all exchanges | 13,178 |
| Exact unique ticker-to-CIK matches | 2,708 |
| Unmatched listing symbols | 462 |
| Ambiguous ticker-row matches / invalid matched CIKs | 0 / 0 |
| Matched rows reporting NYSE / empty exchange | 2,699 / 9 |
| Distinct CIKs among matches | 2,618 |
| CIK groups shared by multiple matched NYSE tickers | 60 |
| Matched NYSE rows in those shared-CIK groups | 150 |

Examples actually present: UNM/UNMA share CIK 0000005513; BDX/BDXB share 0000010795; DDS/DDT share 0000028917. This report does not infer security types from those ticker pairs. Shared issuer relationships are legitimate and are not necessarily source errors.

Seven assertion groups passed: mapping publication precedes listing publication; both raw blob identities match; NYSE symbols are unique; join partitions reconcile; exchange partitions reconcile; issuer-group row counts reconcile; and reported multi-ticker groups contain actual shared keys. These are data-integrity checks only. The [JSON artifact](../learning/2026-11-17-nyse-issuer-join-audit.json) preserves source URLs, commit times, bytes, raw SHA-256 and Git blob fingerprints.

To reproduce: fetch the artifact's two immutable URLs, verify SHA-256 and Git blob hash (SHA-1 of the Git blob header plus exact bytes), parse JSON rows and CSV without numeric CIK conversion, index the CSV by exact Ticker, and partition each NYSE symbol into missing, ambiguous, invalid-CIK or unique match. Group unique matches by CIK and reconcile the totals above. This checks only one historical snapshot.

## Evidence change and next action

The identity candidate gains measured historical issuer coverage. It loses support for using CIK as the sole permanent **security** identifier: 150 matched listings would share keys with another matched listing. None of the 2,708 matches is established as an eligible common stock merely by this join.

Do not discard the 462 unmatched names, carry a present-day mapping backward, or treat missing exchange as NYSE. Any symbol-normalization policy must be separately specified and audited; no return-driven normalization is permitted. There are no delisting-price, corporate-action, quote or settlement observations in this diagnostic.

Next bounded task: inspect security-registration tables in filings publicly available before this historical cutoff for the three demonstrated shared-CIK pairs, recording distinct security titles, source acceptance times and any unresolved successor relationships. A versioned issuer/class/ticker identity can be proposed only from that evidence. Then audit a fixed interval's matching total-return and delisting-price coverage before evaluating the existing two-arm momentum/buffer proposal. Keep its parameters fixed; no extra dropout/holding-period variants.

No order, broker connection, model tag, risk, promotion, workflow, secret, cloud or deployment change. No data purchase, account creation, merge or training occurred.
