# Continuous v3 research loop

The v3 research phase now has a persistent controller:
`scripts/run_v3_research_controller.py`.

It evaluates the existing causal BTC/ETH v3 model at both `$4,000` and
`$6,000` order notionals. It records each report and keeps a small state file
so an unchanged dataset is not backtested repeatedly. A new historical data
fingerprint or a change to the research implementation starts the next
iteration.

The controller marks `strategy_ready: true` only when the same candidate passes
at both sizes and all of these conditions hold:

- the v3 base-cost and higher-cost walk-forward medians are positive;
- the full sample is profitable and has at least eight entries;
- every base and stress walk-forward fold has at least five entries;
- no full sample or fold has a permanent halt or repeated kill-switch events.

The 1-day, 1-week, 1-month, and 1-year results remain visible in each report.
The 1-day snapshot is evidence only; it cannot make a strategy ready by
itself. A passing result is still a research finding requiring human review,
paper probation, and explicit approval. The controller never changes the
active profile, stages a finalist, enables leverage, or places orders.

## Run once

From the repository root, with the three-year normalized BTC/ETH files present:

```bash
python scripts/run_v3_research_controller.py \
  --btc-path data/historical/binance/normalized/BTCUSDT_1h.csv \
  --eth-path data/historical/binance/normalized/ETHUSDT_1h.csv \
  --output-dir artifacts/momentum-v3/research-controller
```

The current state is written to:
`artifacts/momentum-v3/research-controller/state.json` and
`artifacts/momentum-v3/research-controller/latest.json`.

## Keep a local process running

This waits between checks and stops when the readiness gate passes:

```bash
python scripts/run_v3_research_controller.py \
  --until-ready \
  --interval-hours 24 \
  --btc-path data/historical/binance/normalized/BTCUSDT_1h.csv \
  --eth-path data/historical/binance/normalized/ETHUSDT_1h.csv \
  --output-dir artifacts/momentum-v3/research-controller
```

Use `--force` for an intentional rerun when the files and research code are
unchanged. For an unattended repository runner, the included
`.github/workflows/momentum-v3-research.yml` runs after each completed month,
downloads public Binance Vision history, uploads the state and reports, and
uses read-only repository permissions.

Do not use this loop as permission to promote a model or start live trading.
