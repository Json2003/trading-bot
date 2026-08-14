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

## Context, liquidity, and execution gates

The runner accepts an optional operator-supplied context CSV with the columns
`timestamp,sentiment,impact`. Sentiment must be between -1 and 1; impact must
be non-negative. Events are aligned strictly as-of each bar: future events are
never visible, and events older than the configured lookback expire. If no
context CSV is supplied, sentiment is neutral and event blocking is inactive;
the causal liquidity and execution-cost gates still apply.

Entries also require volume relative to the prior-bar median, an ATR-based
expected move that covers the configured round-trip fees and slippage plus a
minimum edge buffer, and the existing higher-timeframe/regime/leader
conditions. Strongly adverse context reduces size; high-impact or sufficiently
negative context blocks entries. The runner keeps one position at a time, so
the portfolio exposure/correlation control remains explicit. These are
research filters only and do not change live or paper risk limits.

For a context-enabled run:

    python scripts/run_v3_research_controller.py \
      --btc-path data/historical/binance/normalized/BTCUSDT_1h.csv \
      --eth-path data/historical/binance/normalized/ETHUSDT_1h.csv \
      --context-csv data/research/context.csv \
      --output-dir artifacts/momentum-v3/research-controller

The controller fingerprints the context file as part of the experiment input,
so changing it causes a fresh research iteration rather than reusing prior
evidence. No network sentiment feed is fetched by the runner; any context file
must be supplied as a timestamped research input.

## Matrix diagnostic on repository sample data

The repository includes a 240-row synthetic OHLCV fixture for smoke testing the
matrix only. It is not market evidence and must not be reported as strategy
performance. Run it with a window size that produces three non-overlapping
windows:

    mkdir -p artifacts
    python scripts/run_research_matrix.py \
      backtest/sample_data/sample_ohlcv.csv \
      --window-size 80 \
      --test-fraction 0.30 \
      --spread-bps 12 \
      --slippage-bps 8 \
      --commission-bps 0 \
      --expected-move-bps 40 \
      | tee artifacts/research-matrix-sample.json

The matrix applies execution costs and reports net return, cost drag, drawdown,
Sharpe, profit factor, and trade count. A family is not research-gate eligible
because its best test slice is profitable: it needs at least three windows,
at least five trades per window, positive median net return and Sharpe, a finite
median profit factor of at least 1.05, at least two-thirds profitable windows,
and no test drawdown above 20%. A failed gate is expected on a small synthetic
fixture.

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
