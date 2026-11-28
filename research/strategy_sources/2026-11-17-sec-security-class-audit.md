# Which of six historical NYSE listings were common stock?

At the January 1, 2022 00:11 UTC listing snapshot, **UNM, BDX and DDS pass the instrument-type screen**. UNMA, BDXB and DDT each have documentary evidence of a different security type. These findings cover the three pairs selected in the [preceding issuer audit](2026-11-17-historical-issuer-identity-audit.md). The other 3,164 listing strings were outside this review, including all 462 with unresolved issuer matches.

| Company | Common stock identified | Exclude from the common-stock universe | Primary evidence available by the snapshot |
| --- | --- | --- | --- |
| Unum Group | UNM, common stock | UNMA, 6.25% junior subordinated notes due 2058 | [Quarterly filing](https://www.sec.gov/Archives/edgar/data/5513/000000551321000151/unm-20210930.htm) |
| Becton, Dickinson | BDX, common stock | BDXB, depositary shares representing a one-twentieth interest in the 6% Series B mandatory convertible preferred | [Annual filing](https://www.sec.gov/Archives/edgar/data/10795/000001079521000091/bdx-20210930.htm) |
| Dillard’s | DDS, Class A common stock | DDT, 7.5% capital securities issued by Dillard’s Capital Trust I | [Quarterly filing](https://www.sec.gov/Archives/edgar/data/28917/000002891721000246/dds-20211030.htm), [trust prospectus](https://www.sec.gov/Archives/edgar/data/1066225/0000930661-98-001661.txt), [ownership filing](https://www.sec.gov/Archives/edgar/data/28917/000002891708000049/edgar.xml) |

## How to handle DDT and historical changes

The December 2021 Dillard’s quarterly cover lists only DDS. A matching ticker on the parent’s cover would therefore miss DDT. The earlier ownership filing explicitly names DDT as a trust capital security. The trust prospectus identifies its legal issuer as CIK 0001066225 and the parent as guarantor. The parent’s March 2021 annual filing describes the trust securities then outstanding. At the snapshot the GitHub listing names the trust, while the SEC-derived ticker map associates DDT with the parent’s CIK 0000028917. These sources together support the class exclusion. The trust’s original application for listing alone would not establish continuing exchange membership. [Annual filing, note 7](https://www.sec.gov/Archives/edgar/data/28917/000002891721000066/dds-20210130.htm)

Preserve the source ticker, mapped filer, issuing entity, class, exchange evidence, observation time and accession separately. Store successor and effective dates as unknown until a dated event establishes them. BDXB’s filing describes a scheduled conversion by June 1, 2023. It supplies no realized conversion terms as of this snapshot. Splicing BDXB onto BDX would introduce unsupported prices and holdings. Unmatched or ambiguous instruments remain visible in coverage counts.

## One corporate action that can expose a bad price feed

Dillard’s declared a **$15 special dividend per common share**. For DDS, OCC’s contemporaneous memo gives November 26, 2021 as the ex-date, November 29 as the record date and December 15 as the payable date. The issuer’s quarterly filing confirms the amount and the affected Class A and B common stock. This gives a dated check on any candidate price and corporate-action feed. The equity ledger should recognize the entitlement once, keep cash unavailable until payable and avoid assigning this dividend to DDT. Actual payment receipt and fill prices were outside this audit. [Issuer filing, note 15](https://www.sec.gov/Archives/edgar/data/28917/000002891721000246/dds-20211030.htm) · [OCC memo 49621, November 19, 2021](https://infomemo.theocc.com/infomemos?number=49621)

## What was verified

The sample, snapshot and exact-symbol rule came from the [prior machine-readable join](../learning/2026-11-17-nyse-issuer-join-audit.json). No price outcomes selected the names. Both GitHub inputs matched their prior byte fingerprints. Each of the six symbols had one exact NYSE listing and one exact mapper row with the expected CIK and exchange.

The [offline extractor](../learning/tools/sec_cover_rows.py) reads the displayed inline-XBRL security title, ticker and exchange in the same source context. The three covers supplied 2, 11 and 1 such rows, respectively. Five of the six target tickers had a direct cover match. DDT required the separate evidence above. A diagnostic that moved one Unum title to a different context was rejected. Missing cover facts require further evidence. An empty result cannot establish a delisting. The utility supports the filing format tested here and leaves classification and time eligibility to the audit.

| Filing | SEC acceptance in UTC |
| --- | --- |
| Unum 10-Q | 2021-11-03 20:17:15 |
| Becton, Dickinson 10-K | 2021-11-24 19:57:19 |
| Dillard’s 10-Q | 2021-12-03 21:30:15 |
| Dillard’s 10-K | 2021-03-29 21:14:08 |
| Dillard’s Form 4, ticker evidence | 2008-10-27 14:10:48 |
| Trust prospectus | Filed August 10, 1998. The archival header gives day precision. |

The [evidence file](../learning/2026-11-17-sec-class-evidence.json) records source URLs, accessions, filing times, original Git revisions, retrieved byte sizes and hashes, class decisions and checks. The historical cutoff applies to information used for the classifications. The research retrieval time is recorded separately. Acceptance time supplies a filing-time bound. No intraday delivery time was inferred. Underlying listing files, regulatory filings and vendor data are not republished.

To repeat the cover extraction with Python 3, save a primary filing from the evidence file locally and run the following command from the repository root.

```sh
python research/learning/tools/sec_cover_rows.py unm-20210930.htm
```

Compare the displayed values and context IDs with the recorded rows. To reproduce a semantic SHA-256, serialize the returned list in its emitted order with Python `json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()` and hash those bytes. The separate raw hashes identify the responses received. SEC-hosted HTML may contain a changing delivery wrapper.

## Consequence for the momentum comparison

One historical security-class audit was completed. Strategy trials, training runs and unseen confirmations were zero. Net return, drawdown, Sharpe, profit factor, fills, turnover and realized costs were unmeasured. The hold-buffer hypothesis gained no performance evidence. The existing research contract remains $5,000 settled cash, whole shares, $1,000 per position and 43 bps of stress on each traded side.

The next bounded task is a **coverage audit for January 3–31, 2022**, the first trading month after this snapshot. Retain the starting denominator and track dated changes, gaps and exits, including any unresolved names. Check dated instrument IDs against unadjusted opens, opening bid/ask and size, splits, cash dividends and delisting proceeds. Reconcile the DDS special dividend separately as a known pre-period input to momentum history. Report missing coverage by field and session before calculating returns. NYSE documents a consolidated historical source with trades, NBBO and a security master. Its public Daily TAQ sample index showed no January 2022 files when inspected. Access to a continuous January feed remains unverified. [NYSE product](https://www.nyse.com/data-products/catalog/daily-taq/) · [Public sample index](https://ftp.nyse.com/Historical%20Data%20Samples/DAILY%20TAQ/)

This fixed month is a data diagnostic. The strategy comparison still needs adequate signal lookback, a frozen evaluation calendar, the existing independence gates and unseen confirmation.
