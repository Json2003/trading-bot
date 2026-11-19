**Offline NYSE research runner**

Run the implemented top-five momentum and retain-through-rank-ten comparison with one command from a research checkout containing this change.

```sh
python scripts/run_nyse_offline_research.py --manifest data/nyse/manifest.json
```

The command reads local files, validates their hashes and writes an immutable run directory below artifacts/nyse-offline. Open summary.md for the owner review and report.json for the full rankings, eligibility decisions, planned orders, ledger, daily cash/equity and metrics. summary.json provides a compact machine-readable result. Each run saves its specification and manifest. Source files remain at their original locations and must be retained for reproduction. report.json records the original absolute manifest and input paths. The saved manifest is a provenance copy. Repeat the command with the original manifest beside its versioned files. The code and data content hashes determine the run ID. Repeating the same input publishes or reuses the same artifact. Atomic directory publication prevents a concurrent retry from exposing a partial report.

To build inputs from public market snapshots, use the [forward capture and preparation commands](nyse-forward-capture.md). Add `--check-inputs` to validate a finished manifest before either arm runs. The check requires completed bars for the eligible cohort and preserves a separate run identity.

The frozen specification is research/experiments/nyse_momentum_retention_v1.json. Monetary values use decimal strings. Initial cash is $5,000, the entry-notional cap is $1,000 and stress costs are 43 bps per buy or sale. Both arms share the source scores, cash rules, a stop-distance sizing budget of 0.5% of equity before stress fees, two-ATR stop, prior-volume participation of 0.1% and 2% close-observed drawdown latch. These risk values come from PR #290's NYSE diagnostic. Monthly selection supplies the new experiment's holding rule. The earlier SMA diagnostic's twelve-session time exit belongs to that strategy. A marked position can exceed its entry cap after a price change, triggering a reduction planned at the next decision and filled at the following open.

At each month-end decision, rank eligible securities by momentum_close[t-21] / momentum_close[t-252] - 1 and break ties by permanent security ID. Both observations use the supplied session calendar. A 252-session lag requires 253 observations through t. Eligibility requires a documented NYSE common-stock identity, raw close above $5 and a twenty-session median raw dollar volume of at least $20 million. All 253 required history rows must exist and be available at the decision. Unresolved classification or missing eligible history blocks the run. Explicit exclusions are retained in each decision's eligibility records.

The naive arm chooses the top five. The buffered arm keeps existing holdings while their ranks are at most ten and fills vacancies in rank order. Existing quantities persist through ordinary retention. Cash shortages defer vacancies, with new whole-share intentions planned after subsequent closes against the already selected monthly targets. Sells settle on the applicable T+2 or T+1 calendar. Purchases reserve fees and consume settled cash only. An adverse opening gap that violates cash or the position cap rejects the previously planned quantity. Rank-based exits, cap reductions and a latched account generate following-open sales. Stops include the entry bar and adverse gaps. A stopped security becomes eligible for a new entry at the next monthly selection. Ending positions and claims are marked without an invented final-close sale.

Execution uses raw next-session opens and adverse daily-bar stop proxies. These assumptions make the v1 output a development diagnostic. Verified order-time quote prices, sizes, halts and actual fills still require an execution-data adapter. Receipts use the supplied payable dates as explicit research assumptions. An undated cash claim remains unavailable. Fractional shares are valued as entitlements to the successor security, then converted to a cash receivable only when a separate payment-price event becomes available. Corporate exchanges do not receive an invented sale commission. Both market and synthetic runs retain research-only flags, and the runner has no broker or model-training dependency.

The manifest uses this shape. The hashes below are placeholders to replace with SHA-256 of the exact local files. The example dates illustrate the contract and are not a completed market evaluation.

```json
{
  "schema_version": 1,
  "dataset_id": "owner-selected-versioned-dataset",
  "data_kind": "market",
  "evidence_role": "development_diagnostic",
  "signal_price_basis": "asof_total_return_index",
  "universe_selection": "Describe the universe selected before inspecting outcomes",
  "universe": ["permanent-security-id"],
  "blocks": [{"name": "development", "start": "2026-01-02", "end": "2026-01-30"}],
  "corporate_actions_complete_through": "2026-01-30",
  "membership_complete_through": "2026-01-30",
  "source_evidence": {
    "capture_plan_sha256": "REPLACE",
    "input_review_sha256": "REPLACE",
    "capture_receipts": [{
      "file": "capture.json",
      "sha256": "REPLACE",
      "revision": "PINNED_SOURCE_REVISION",
      "from": "2025-01-02",
      "through": "2026-01-30",
      "partitions": [{"url": "SOURCE_QUERY", "received_at": "2026-01-31T00:00:00Z", "sha256": "REPLACE"}]
    }]
  },
  "files": {
    "calendar": {"path": "calendar.json", "sha256": "REPLACE", "source": "SOURCE", "revision": "VERSION", "received_at": "2026-11-18T00:00:00Z"},
    "securities": {"path": "securities.json", "sha256": "REPLACE", "source": "SOURCE", "revision": "VERSION", "received_at": "2026-11-18T00:00:00Z"},
    "prices": {"path": "prices.json", "sha256": "REPLACE", "source": "SOURCE", "revision": "VERSION", "received_at": "2026-11-18T00:00:00Z"},
    "actions": {"path": "actions.json", "sha256": "REPLACE", "source": "SOURCE", "revision": "VERSION", "received_at": "2026-11-18T00:00:00Z"}
  }
}
```

Each input is a JSON array. File paths resolve relative to the manifest. Receipt time records acquisition of the file. Row available_at records when the information became usable for the historical decision. For market inputs, source_evidence makes the frozen plan, reviewed membership/action packet and raw capture receipts part of the immutable manifest identity. A newly retrieved historical file cannot establish its original availability without further evidence. Source completeness declarations require a documented independent coverage audit. A permanent security ID must represent one security class. A filer CIK alone may join several instruments.

| File | Required row fields | Conventions |
| --- | --- | --- |
| calendar | session, open_at, close_at, decision_at | Ordered unique session dates, timezone-aware timestamps, NYSE core open at 09:30 New York time. Decision follows close and precedes the next open. Include 253 prior observations and two post-window settlement sessions. |
| securities | security_id, ticker, exchange, instrument_type, currency, eligible_from, eligible_until, available_at | Exchange XNYS and type COMMON_STOCK identify eligible entries. USD currency. eligible_until is exclusive or null. Include mandatory merger successors even when they are outside the entry universe. |
| prices | security_id, session, open, high, low, close, volume, momentum_close, atr14, available_at | Raw OHLC in the share units traded on that date. Positive integer volume. momentum_close is a separately audited total-return index using actions admissible at that date. ATR14 is in the raw share units of the completed session. ATR can be null during warmup and must be positive when an entry is planned. Full OHLC becomes available at or after session close. |
| actions | action_id, kind, security_id, session, available_at, plus fields below | Unique event IDs. Supply events in explicit precedence for the same opening session. Record the event on the first trading session it affects. Event terms must be available by that open. |

| Action kind | Additional fields | Accounting |
| --- | --- | --- |
| dividend | amount, payable_date | Accrue for shares held before the ex-session. Dividend income enters equity once. Payment date is a research receipt assumption and may be null. |
| split | new_shares, old_shares | Apply the exact ratio and adjust the raw stop level. Separate any fraction, preserving allocated basis and income. SLG's ratio is new_shares 1 and old_shares 1.03060. |
| stock_merger | new_shares, old_shares, successor_id | Carry the ratio to the successor ID and keep any fraction separate. CIT's common-share ratio is 0.06200 new to 1 old. Mandatory successor whole shares are queued for following-open exit. |
| cash_merger | amount, payable_date | Replace the position with the legal cash entitlement. Unknown receipt date retains the claim as unavailable cash. Casper's common-share consideration was $6.90 gross. |
| fraction_payment | fraction_id, price, payable_date | fraction_id refers to the original split or stock-merger action_id. The supplied price determines cash in lieu when the payment-price information becomes available. |
| rename | old_ticker, new_ticker | Preserve the permanent ID, quantity, basis and stop. No sale or repurchase. |

The first version explicitly rejects merger consolidation into an already held successor and a new share transformation on an unresolved fractional entitlement. Those cases need additional lot reconciliation. Suspensions without a supported security event and uncertain daily-bar execution remain source-audit requirements. Full market-wide membership reconstruction, neural inference and broker connectivity are separate work.

All comparison metrics are computed after stress costs. Net return and drawdown are fractions, with drawdown negative. Costs are USD and turnover is traded notional divided by starting cash. Sharpe uses daily portfolio returns, sample standard deviation, zero risk-free rate and sqrt(252) annualization. Profit factor uses realized economic gains and losses including allocated accrued dividend income, and remains null when there are no realized losses. The account equity includes remaining positions and receivables. Closed-sale count, total simulated fills and realization records are recorded separately. The independent sample count is capped by both PR #290's nonoverlapping-interval count and the number of closed monthly entry cohorts. The existing minimum of twenty applies. Monthly cohort observation IDs let later reviews identify overlap between reruns.

A synthetic run reports software_fixture_complete and zero investment trials. A completed market input reports historical_diagnostic_complete. Blocked input reports blocked_market_data, null metrics and the precise failure, with exit code 2. A completed diagnostic returns exit code 0 even when returns or sample gates fail. Every arm records those failures. The result adapter canonicalize_nyse maps one completed market arm/block to the existing governed schema with explicit units. It rejects synthetic and blocked runs. The command itself does not advance the market checkpoint. Reproductions and overlapping windows retain their original observation identity.

Verify the implementation locally.

```sh
python -m unittest discover -s tests -p 'test_nyse_*.py' -v
```

The tests generate synthetic inputs, invoke the actual CLI, compare both portfolio arms, replay concurrent retries and exercise the audited corporate-action terms with assumed test prices and receipts. These checks measure software correctness. The existing January data audit remains the source of the market-data gaps that block a genuine comparison. [January coverage audit](../strategy_sources/2026-11-17-january-2022-coverage.md)
