# Zero-Cost Launch and Premium Upgrade Plan

## Goal

Launch a reliable paper-trading desktop application without monthly hosting,
market-data or AI-service costs. Preserve clean interfaces so paid services can
be introduced later without rewriting strategy, execution, risk or frontend
code.

## Product decision

The initial product is a **local Windows desktop application**, not a public web
service.

```text
Trading Bot Operator.exe
  -> secure Electron IPC bridge
  -> FastAPI on 127.0.0.1 only
  -> TradingOperatorService
  -> strategy engine
  -> broker adapter
  -> SQLite state and audit log (next milestone)
```

This avoids cloud hosting, domain, TLS, database and managed-worker charges. It
also keeps broker credentials on the trading computer.

## Phase 0 — Rescue smoke release: $0/month

Purpose: prove that installation, startup, strategy cycles, execution, positions,
controls and failure handling work end to end.

Included:

- Windows Electron operator console.
- One-command managed launcher.
- Loopback-only authenticated API.
- Synthetic replay market data.
- In-memory paper broker.
- Existing deterministic strategy suite.
- Start, pause, stop-and-cancel, cancel-all and emergency-stop controls.
- OpenClaw-compatible operator endpoints.
- GitHub Actions validation and Windows installer artifact.

Not included:

- Real market data.
- Real broker connectivity.
- Public internet exposure.
- Live trading.
- Arbitrary manual order entry.
- Automatic kill-switch reset.

Exit criteria:

1. Windows installer builds successfully.
2. A fresh Windows installation launches with one command.
3. The engine completes at least 1,000 synthetic cycles without an unhandled
   fault.
4. Pause prevents additional cycles.
5. Stop cancels all open orders.
6. Emergency stop latches and survives until manual recovery.
7. Closing the desktop application stops the managed API process.

## Phase 1 — Free real-market paper beta: $0/month

Purpose: replace synthetic data and fills with a real broker paper environment
while retaining the same operator and safety boundaries.

Recommended first provider: Alpaca Paper Trading.

Why:

- Paper-only accounts are available without funding.
- Paper and live APIs share the same general contract.
- Free IEX market data is sufficient for functional testing.
- Premarket and after-hours paper testing is supported.

Required work:

- Implement `AlpacaPaperBroker` against the canonical broker protocol.
- Implement account, order, fill, position and market-clock adapters.
- Add websocket reconnect and heartbeat handling.
- Persist broker events and idempotency keys in SQLite.
- Reconcile local state against the broker at startup and periodically.
- Add stale-data, spread, buying-power and market-hours gates before submission.
- Add a provider capability object so IEX limitations are visible to strategy
  code and the dashboard.
- Set the simulated paper balance to $2,000 for realistic strategy testing.

Exit criteria:

1. Submit, acknowledge, partially fill, fill and cancel paper orders.
2. Disconnect during an open order and recover without duplicate submission.
3. Restart the application and reconstruct orders and positions correctly.
4. Complete 20 market sessions in paper mode without an unreconciled incident.
5. Complete at least 100 paper trades under the proposed $2,000 risk rules.

## Phase 2 — Free private remote operation: $0/month

Purpose: operate and observe the bot from a phone without opening a public port.

Recommended approach:

- Keep the API bound to loopback by default.
- Add an optional Tailscale-sidecar mode that binds only to the machine's
  Tailscale interface after explicit configuration.
- Keep bearer authentication and device ACLs.
- Let OpenClaw run on the same trading computer and call the operator API
  locally.
- Use OpenClaw only for alerts, reports and approved lifecycle controls.

Do not:

- Port-forward the API from the home router.
- Store broker keys in OpenClaw prompts or skills.
- expose arbitrary order submission or kill-switch reset.

## Phase 3 — First paid upgrades

Buy services only when a measured limitation is blocking performance or
reliability.

### Upgrade A: better stock data

Trigger:

- IEX coverage misses relevant small-cap volume or quote behavior.
- The scanner needs consolidated quotes, trades or faster snapshots.
- Backtesting requires more history or bulk files.

Adapter:

```text
MarketDataProvider
  get_snapshot(symbol)
  stream_quotes(symbols)
  stream_trades(symbols)
  get_bars(symbol, timeframe, start, end)
  get_reference(symbol)
  health()
```

Initial free option:

- A free end-of-day/historical tier for research and scanner development.

Paid progression, based on measured need:

1. Delayed/unlimited aggregate plan.
2. Historical trades and deeper backtest plan.
3. Real-time consolidated quotes/trades only when paper results justify it.

No strategy should import a vendor SDK directly. Vendor code belongs under
`market_data/providers/`.

### Upgrade B: news and catalysts

Trigger:

- False scanner candidates are primarily caused by missing catalyst data.

Add a `CatalystProvider` interface and begin with manually curated or public
company/SEC events. Purchase real-time news only after logging proves catalyst
classification is the bottleneck.

### Upgrade C: monitoring and error reporting

Trigger:

- The app is running unattended or on more than one machine.

Start free/local:

- Structured JSON logs.
- SQLite incident table.
- Rotating local log files.
- OpenClaw notifications.

Paid later:

- Hosted error reporting.
- Long-term metrics and log retention.
- Managed uptime checks.

### Upgrade D: cloud runtime

Trigger:

- Home internet or power interruptions materially affect paper results.
- The user needs continuous availability away from the trading computer.

Before cloud deployment:

- Containerize only the rescued operator/engine service, not legacy `server.py`.
- Encrypt secrets through the cloud provider's secret manager.
- Use a private network; do not expose broker controls publicly.
- Add durable database backups and process supervision.
- Estimate total monthly data egress, compute and storage before committing.

### Upgrade E: AI and OpenClaw enhancements

Trigger:

- Deterministic reports and alerts are stable, but interpretation consumes too
  much time.

Allowed AI uses:

- Summarize daily activity.
- Explain deterministic strategy decisions from stored features.
- Classify operational incidents.
- Rank scanner candidates after deterministic hard filters.

Disallowed AI uses:

- Bypass position sizing or risk gates.
- Invent arbitrary symbols/orders outside the strategy contract.
- Reset a kill switch.
- Modify live risk settings without a separate authenticated workflow.

## Cost-control rules

1. Default every optional integration to disabled.
2. Require an explicit environment variable to enable a paid provider.
3. Log provider request counts and estimated cost.
4. Add monthly request ceilings in configuration.
5. Fail closed when a premium entitlement expires.
6. Keep a working free fallback for research, reports and synthetic tests.
7. Do not buy a service until a written experiment shows the expected benefit.

## Suggested repository interfaces

```text
tradingbot_ibkr/
  providers/
    market_data.py
    catalysts.py
    notifications.py
  execution/
    broker_base.py
    paper_broker.py
    alpaca_paper_broker.py
  persistence/
    sqlite_store.py
  runtime/
    synthetic.py
    alpaca_paper.py
```

Provider selection should occur in one runtime factory:

```python
runtime = build_runtime(settings)
```

The frontend and OpenClaw integration should not know which broker or data vendor
is active. They consume only operator status, orders, positions, incidents and
approved lifecycle commands.

## Funding gates

Do not move to the next spending level based on optimism. Use these gates:

- **$0 -> paid data:** 20 clean paper sessions and evidence that free data is the
  limiting factor.
- **paid data -> tiny live account:** 60 clean paper sessions, restart and
  disconnect drills passed, positive expectancy after realistic slippage, and no
  unresolved high-severity incident.
- **tiny live -> larger capital:** at least 100 live trades, stable execution
  quality, maximum drawdown inside policy and independent reconciliation.
- **local -> cloud:** documented downtime cost exceeds expected hosting cost.

## Immediate rescue sequence

1. Make Python CI and Windows packaging green.
2. Produce the first downloadable Windows installer artifact.
3. Add SQLite persistence and a latched incident record.
4. Add restart reconciliation tests.
5. Implement the Alpaca paper broker adapter.
6. Add the real-market paper runtime behind `TRADING_OPERATOR_RUNTIME=alpaca-paper`.
7. Add the small-cap scanner and opening-range strategy only after the broker and
   market-data path pass acceptance tests.
