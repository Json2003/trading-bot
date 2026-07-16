# Trading Bot Operator Console

The supported rescue frontend is the **Electron desktop operator console** in
`dashboard/electron-app`. It communicates only with the bearer-token-protected
operator API on the local loopback interface.

The older Flask dashboard, standalone simulation screen and Flutter placeholder
remain in the repository as legacy/research material. They are not part of the
packaged application and must not be used to control a broker.

## Supported architecture

```text
Electron renderer
  -> fixed contextBridge methods
Electron main process
  -> bearer token held only in process environment
http://127.0.0.1:8765
  -> paper-only TradingOperatorService
  -> configured deterministic strategy engine
  -> canonical PaperBroker execution contract
```

The renderer has no direct network access, no token storage and no arbitrary
order-entry API. Available controls are limited to status, positions, orders,
start paper mode, pause, stop-and-cancel, cancel-all and emergency stop.

## One-command local launch

Requirements:

- Python 3.10 or newer
- Node.js with `npm`
- Project Python dependencies installed (`pip install -e .[dev]`)

From the repository root:

```bash
python scripts/launch_operator_console.py
```

The launcher:

1. Generates an in-memory session token unless one is already configured.
2. Starts the operator API on `127.0.0.1:8765`.
3. Waits for the health endpoint.
4. Runs `npm ci` when Electron dependencies are missing.
5. Opens the desktop console.
6. Stops the API when the desktop app closes.

The default engine is `synthetic-multi-strategy-smoke`. It exercises strategy,
execution, paper fills, positions and operator controls without broker
credentials. It is not a live-market strategy and does not demonstrate
profitability.

Start the interface without an engine for diagnostics:

```bash
python scripts/launch_operator_console.py --runtime none
```

In that mode the API remains observable, but the Start button is disabled.

## Manual launch

Set one long random token in both processes:

```bash
export TRADING_OPERATOR_TOKEN='replace-with-a-long-random-value'
python scripts/run_operator_api.py
```

Then in another terminal:

```bash
cd dashboard/electron-app
npm ci
npm start
```

PowerShell uses `$env:TRADING_OPERATOR_TOKEN = '...'` instead of `export`.

## Validation and packaging

```bash
cd dashboard/electron-app
npm ci
npm run check
npm test
npm run dist
```

`npm run dist` creates the current platform installer under `dist/`. GitHub
Actions workflow `Desktop Operator` validates the renderer contract and builds a
Windows NSIS installer artifact.

## Security rules

- Do not bind the operator API to a non-loopback address.
- Do not place broker credentials in Electron, OpenClaw or renderer storage.
- Do not add arbitrary order submission, live activation, risk editing or
  kill-switch reset to the operator API.
- A latched kill switch requires recovery outside the operator interface.
- Live broker deployment remains blocked until reconciliation, restart recovery,
  pre-trade risk ordering and broker-paper integration pass their acceptance
  tests.
