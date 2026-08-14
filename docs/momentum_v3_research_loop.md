# Continuous v3 research loop

The v3 research phase is driven by the persistent controller at
scripts/run_v3_research_controller.py. It is a bounded research process, not a
promotion or trading process.

Every iteration evaluates exactly two order notionals: $4,000 and $6,000.
Readiness requires the same candidate to pass at both sizes. Each size must
contain:

- a profitable finite full-sample result at base and stress costs;
- exactly three walk-forward folds, with finite results and at least five
  entries in every base and stress fold;
- a profitable finite one-year confirmation holdout at both cost levels, with
  at least five entries and no permanent halt or repeated kill-switch event;
- visible evidence for the 1-day, 1-week, 1-month, and 1-year snapshots.

The four horizon snapshots are evidence only. In particular, a positive 1-day
result cannot make a candidate ready. The full sample, stress full sample,
walk-forward medians, and confirmation holdout are the actual gate inputs.
Stress costs must be finite, non-negative, and strictly higher than base costs.

The controller uses fixed candidate definitions from the research runner. It
does not select a winner by taking the best result from an arbitrary search
space. A code or cost/configuration change changes the experiment signature and
resets the readiness streak. The controller also requires the same candidate
to pass at both sizes on three distinct data vintages before recording
strategy_ready. These vintages are deliberately treated as repeated evidence,
not as independent proof; a human must still review the reports and holdout
before any promotion decision.

The output state is research metadata only. The controller never changes the
active profile, stages a finalist, enables leverage, places paper or live
orders, changes risk limits, or merges a pull request. Any promotion remains a
separate manual action after review and paper probation.

## Run once

From the repository root, with normalized BTC and ETH files present:

    python scripts/run_v3_research_controller.py \
      --btc-path data/historical/binance/normalized/BTCUSDT_1h.csv \
      --eth-path data/historical/binance/normalized/ETHUSDT_1h.csv \
      --output-dir artifacts/momentum-v3/research-controller

The controller rejects non-finite or malformed OHLCV values, mismatched BTC/ETH
coverage, unsynchronized gaps, gaps over six hours, and data older than 45 days
by default. It writes state.json, latest.json, and per-iteration reports under
artifacts/momentum-v3/research-controller. An old state schema is discarded
rather than trusted.

## Unchanged data and bounded polling

When the input data, source files, and research configuration have not changed,
the controller records a skip and does not rerun the backtest. Use --force only
for a deliberate rerun. The --until-ready mode requires --max-iterations so an
unattended process has a finite stop bound, for example:

    python scripts/run_v3_research_controller.py \
      --until-ready \
      --max-iterations 30 \
      --interval-hours 24 \
      --btc-path data/historical/binance/normalized/BTCUSDT_1h.csv \
      --eth-path data/historical/binance/normalized/ETHUSDT_1h.csv \
      --output-dir artifacts/momentum-v3/research-controller

The monthly GitHub Actions workflow downloads completed Binance Vision months,
restores the most recent successful research artifact when available, and
uploads the new state and reports. Its token has only actions: read and
contents: read permissions. Restore is optional; stale or incompatible state
is never used as evidence.

Do not use this loop as permission to promote a model or start live trading.
