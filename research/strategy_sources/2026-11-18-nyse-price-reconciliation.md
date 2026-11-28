# NYSE price and action reconciliation — November 18, 2026

**Price cross-check completed; reject the retrieved bundle for causal January 2022 evaluation.** The missing records exist in a separately retrieved provider history, and the inconsistent bars have coherent counterparts. Neither source proves original historical availability. No strategy was evaluated. The [reconciliation manifest](../learning/manifests/2026-11-18-nyse-reconciliation.json) preserves results, fingerprints, primary filings, transformations, failures and remaining gaps. The [original audit](2026-11-18-nyse-market-manifest-audit.md) and defective source files were not overwritten.

## Scope and method frozen before comparison

Retain CRM, NOW, SQ, SNOW, UBER, PINS, CHWY, TWLO, PLTR, NET, FSLY and ASAN; December 1, 2020–February 2, 2022; proposed January 3–31 development block. This remains a narrow, retrospectively selected diagnostic, not a representative or survivorship-neutral NYSE universe. No ranks, signals, forecasts, holding-period returns or strategy outcomes were calculated.

The second provider is Yahoo's public chart endpoint, retrieved November 18, 2026. [Yahoo identifies CSI as its US equity historical-data provider](https://help.yahoo.com/kb/SLN2310.html). Dolt's upstream vendor is not established, so **separate provider retrieval is established; full upstream independence is not**. This distinction prevents agreement from becoming two independent confirmations of truth. The original revision remains `uuk0i93qvs5rrk9k8geq250699j4l71c`. Yahoo has no comparable immutable server-side revision; exact response SHA-256 and query URLs are recorded instead.

Two documented present-day source transformations were necessary. They were fixed before the field comparison:

- SQ's endpoint returned HTTP 404. Its successor query symbol XYZ returned the complete historical series. The [issuer's January 21, 2025 SEC exhibit](https://www.sec.gov/Archives/edgar/data/1512673/000162828025007376/exhibit43blockinc2024.htm) establishes the ticker change. Historical SQ remains SQ throughout this experiment. It is not removed, sold or treated as a new 2021 instrument.
- NOW's present historical prices reflect a later five-for-one split. The [issuer's 2025 annual filing](https://www.sec.gov/Archives/edgar/data/1373715/000137371526000007/now-20251231.htm) documents its December 17, 2025 effective date. For field comparison only, multiply Yahoo historical OHLC by five and divide volume by five. No December 2025 action is inserted into the 2021/January 2022 ledger. Agreement with original nominal price levels supports this normalization, not a new predictive feature.

Compare nominal OHLC after cent rounding, allowing at most $0.01 absolute difference per field; require exact share-volume agreement for the separate volume comparison. Price and volume tolerances were fixed before comparison. Monetary thresholds are not used for volume. [Yahoo's adjusted-close documentation](https://help.yahoo.com/kb/SLN28256.html) describes split/dividend adjustment; current adjusted close is not substituted for raw execution prices or certified as a causal signal index.

## Actual results

| Check | Result |
| --- | ---: |
| Second-provider expected / observed cells | 3,552 / 3,552 |
| Second-provider missing / internally invalid bars | 0 / 0 |
| Original missing cells present in second source | 48 / 48 |
| Original seven internally invalid bars with coherent counterparts | 7 / 7 |
| Overlapping opening prices agreeing within $0.01 | 3,504 / 3,504 |
| Overlapping closing prices agreeing within $0.01 | 3,504 / 3,504 |
| High-price disagreements | 11 |
| Low-price disagreements | 1 |
| Exact volume agreements | 36 / 3,504 |
| Nonzero volume differences within 100 shares | 2,978 |
| Volume differences exceeding 100 shares | 490 |
| Volume differences exceeding 1% of the larger reported volume | 23 |

All eleven high disagreements and the one low disagreement occur on May 5, 2021. The seven originally invalid bars have matching opens and closes in the second source but different highs; NOW also has a different low. The four additional high disagreements involve CRM, SQ, CHWY and NET. Thus changing seven opening prices would be the wrong repair. Their opens agree with the separately retrieved history; the daily ranges are disputed. The cause and authoritative consolidated/venue definitions remain unproved.

All twelve stocks have second-provider observations for May 3, May 14, July 7 and July 8, 2021. The original absence remains preserved. Differences in volume are not all classified as errors: most exact differences are small, and provider rounding/venue definitions can differ. The 23 larger discrepancies remain explicit. Volume affects liquidity and participation gates, so do not merge sources blindly or silently substitute one provider's volume.

A Nasdaq public historical endpoint probe returned an invalid-parameter response for slash-formatted dates. The ISO-date retry returned success with zero records for FSLY during May 3–14, 2021. It supplied no third corroborating price series. The failed SQ lookup and both Nasdaq results are retained as failures, not discarded.

## Corporate-action findings

Twelve original issuer annual filings, filed February 3–March 30, 2022, were inspected. Each reports never paying dividends; the detailed evidence retains a short exact excerpt, filing/accession, reported acceptance timestamp, immutable SEC URL and original byte hash. This supports **retrospective absence of cash dividends** for these names in the requested interval. It materially improves on inferring no events from empty vendor tables. These later publications are quality checks, not information claimed available to a January signal.

The second-provider dividend/split event lists are empty in the requested interval. The twelve annual texts contain no stock-split/reverse-split phrase. Absence from these searches is not an authoritative complete action register. No interval-wide complete-through certificate is issued for splits, spin-offs, conversions, terminal consideration or exchange/class continuity. The earlier twelve cover classifications still establish types at filing dates rather than complete permanent-ID histories.

Two issuer-specific confounders were checked directly:

- The [December 1, 2021 Square/Block announcement](https://squareup.com/gb/en/press/square-changes-name-to-block?country_redirection=true) expected the legal name change around December 10, explicitly retained SQ and its CUSIP, and required no action from current holders. Do not create a sale, ticker change or new position for that name announcement.
- The [January 31, 2022 issuer announcement](https://www.sec.gov/Archives/edgar/data/1512673/000119312522023529/d261416dex991.htm) confirms Block acquired Afterpay. The [2021 annual filing](https://www.sec.gov/Archives/edgar/data/1512673/000162828022003825/sq-20211231.htm) describes acquisition consideration/new share issuance. This is not a conversion of existing SQ holders into Afterpay, and no receipt per already held SQ share is inferred.

## Timing decision and next executable action

The original snapshot is dated February 3, after the proposed January decisions, with a timezone-free API clock. The secondary observations were received in November 2026. Neither establishes when its individual records first became available. Determining a commit clock's timezone alone would not prove when that revision was pushed/publicly accessible. No timestamp was backdated to a bar close; no annual filing was moved before its actual publication.

**Reject these retrieved responses as causal January 2022 runner inputs.** The price holes can be located and the conflicting ranges diagnosed, but patching them cannot manufacture original publication history. There is no ready runner manifest and no basis for a net-profit estimate from this bundle. Original artifacts and negative findings remain intact. Current adjustment factors are used to explain vendor histories, not smuggled into a historical cash/whole-share fill model.

A further causal historical run requires source vintages with independently defensible pre-decision availability, documented raw/adjusted OHLC and volume definitions, and a complete class/action ledger. Those inputs are not available in the retrieved endpoints. If they cannot be obtained within the no-purchase/no-account constraints, the useful replacement is a **separately frozen prospective diagnostic** with actual receipt timestamps, documented data-use terms, class-level identity and actions captured before affected decisions. It must retain the same fixed risk contract; daily receipts do not create daily independent monthly investment observations. This task did not launch a prospective collector or alter any schedule.

Yahoo's published terms prohibit redistribution and describe its data as informational. Its raw price series is not copied into Git. This audit retains derived reconciliation counts/flags, URLs and hashes. Public reachability is not permission to deploy it as a trading feed. The current Dolt documentation query exposes a CC BY-SA license, but the historical documentation query did not independently establish historical upstream market-data rights; no rights clearance is claimed.

## Verification and learning accounting

The [offline reconciliation tool](../learning/tools/reconcile_nyse_prices.py) checks the original 307 response fingerprints, all twelve SEC annuals and four primary action-record fingerprints, verifies the quoted dividend text, and compares all 3,552 cells. The [reviewed decision metadata](../learning/manifests/2026-11-18-reconciliation-decisions.json) is separate from calculated counts and cannot replace coverage fields. Two executions using the same frozen raw files produced byte-identical output. Cell counts, field totals, exception sets, dividend-source counts, immutable original manifest identity and blocked status were asserted. Source compilation and whitespace checks passed. These are actual market-data checks, not synthetic profit tests.

```sh
python research/learning/tools/reconcile_nyse_prices.py \
  --secondary RAW_SECONDARY_DIR --original RAW_ORIGINAL_DIR \
  --manifest research/learning/manifests/2026-11-18-nyse-market-data-audit.json \
  --decisions research/learning/manifests/2026-11-18-reconciliation-decisions.json \
  --output reconciliation.json
```

The raw original directory is documented in the prior audit. The secondary directory contains the twelve chart responses (XYZ for historical SQ), `cash-dividend-evidence.json`, `action-primary-index.json`, their SEC/issuer files, and original SEC catalog references. These raw inputs remain outside Git and are transient. Exact public source locators and byte hashes are retained; re-acquisition can detect changed responses. No claim is made that the complete raw bundle was durably redistributed.

One additional source-reconciliation cycle; **zero strategy trials, neural training runs or unseen confirmations**. Net return, drawdown, Sharpe, profit factor, trades, turnover and execution/operating expenses were not measured. No purchases or orders occurred. No strategy gained validated profitability evidence. Confidence improves in identifying the missing sessions, disputed ranges, nominal price levels and dividend absence; the bundle loses support for causal evaluation. Preserve the monthly top-five versus retention comparison, $5,000 cash, whole shares, $1,000 entry cap, 86-bps round-trip stress, crypto pause and all runtime/promotion gates. PR #292 stays draft and unmerged.
