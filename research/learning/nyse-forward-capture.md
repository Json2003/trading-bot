# NYSE market data capture and replay

The November 18 repair adds an end-to-end input path for the offline research runner. Capture stores the original response bytes, the pinned source revision and the UTC time of receipt. Preparation validates those files, applies reviewed corporate actions to the signal series and writes the four hash-checked runner inputs when the entire block is available. The strategy continues to use close-based ranks, whole shares, settled cash and assessed next-session opens.

The [frozen forward plan](../experiments/nyse_prospective_capture_v1.json) covers December 1, 2026 through January 29, 2027. The November 30 close supplies the first ranking. Its cutoff is December 1 at 6 a.m. Pacific, 30 minutes before the NYSE opens. Subsequent session cutoffs use the same next-session time. This accommodates the public daily file arriving overnight. See the [NYSE session calendar](https://ir.theice.com/press/news-details/2025/NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/default.aspx).

Use Python 3.10 or newer in the research checkout. Keep the data directory private and retain its receipt indexes. Restore the versioned captures in a clean checkout, then collect later completed sessions that are missing.

```sh
DATA_DIR=data/nyse-forward
REVIEW=research/learning/manifests/2026-11-19-nyse-forward-input-review.json
for receipt in research/learning/manifests/*-nyse-forward-receipt-index.json; do
  python scripts/prepare_nyse_market_data.py --data-dir "$DATA_DIR" restore --index "$receipt" || exit $?
done
python scripts/prepare_nyse_market_data.py --data-dir "$DATA_DIR" capture
python scripts/prepare_nyse_market_data.py --data-dir "$DATA_DIR" prepare --review "$REVIEW"
```

Restore requires byte-for-byte agreement with the original responses and preserves their receipt times. New captures record actual acquisition times. The command returns `awaiting_source_publication` when requested rows are absent from the pinned revision. Check the returned receipt index even when some rows are still pending. The November 19 receipt contains the full November 18 session and leaves November 19 pending. The daily research task runs at 5 a.m. Pacific starting November 20. Verify the November 30 bars actually arrive by the December 1 decision cutoff at 6 a.m. Pacific.

The baseline review records the class-level evidence and five reconciled events. Extend it with later listing and issuer/event evidence as sessions close. Retain the original class receipt times and record receipt of new action terms before the affected open. Advance the coverage dates only through the sessions actually checked. The current adapter handles stable security classes, cash dividends and splits. An intervening rename, transfer, spin-off or merger requires reconciling its terms and identity before creating a replay manifest. The general runner also understands its documented merger and rename records.

Preparation reports the last contiguous session, eligible price coverage, actionable cutoff, unresolved action or membership dates and any missing bars. The momentum index compounds dividend-inclusive gains in the share units of each session. The raw OHLC fields remain the original nominal inputs. Wilder ATR uses true ranges in the current share units and rescales previous ATR at a split. Output never lets later receipts appear earlier than their recorded times.

The same command returns `ready_for_offline_replay` with the path to its immutable manifest after the block and both reviews are complete. The versioned November 19 review currently covers sessions through November 18. Supply later reviewed evidence as it becomes available. Use the returned manifest path for both commands below.

```sh
python scripts/run_nyse_offline_research.py --manifest "$MANIFEST" --check-inputs
python scripts/run_nyse_offline_research.py --manifest "$MANIFEST"
```

Validation requires every eligible candidate's price history at the first ranking and observed daily bars throughout the requested evaluation block. The runner now rejects incomplete periods even when the portfolio stays in cash. The input-only run records validation and zero strategy trials. The replay records both arms, ledger reconciliation, modeled expenses, net return, drawdown, Sharpe, profit factor, fills and independent cohorts. The 20-observation requirement remains in force. The two-month window cannot meet that gate by itself, and daily-bar execution remains a proxy.

The [source and readiness audit](../strategy_sources/2026-11-18-nyse-forward-data-repair.md) describes the actual capture, source comparisons and scope. The [offline runner contract](nyse-offline-runner.md) lists the manifest schema and accounting conventions.
