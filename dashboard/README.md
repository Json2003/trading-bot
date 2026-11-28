# RomanBot Workstation

The browser and Electron desktop app share the same NYSE workstation. Open
the Overview for candles, volume and a three-stock watchlist (IBM, JPM, KO).
Positions, Orders, Research Lab and TWS Connection each have their own view.
The supported browser UI is supplied by the local operator; the older Flask,
standalone dashboard, PyWebView and Flutter screens remain legacy interfaces.

| Workspace | Data | Available actions |
| --- | --- | --- |
| Simulator | Synthetic NYSE stock bars and hypothetical fills | Start, pause, stop and cancel, cancel open orders, emergency stop |
| IBKR Paper | Verified paper account in a local TWS session | Check expected account, view balances and positions, request historical stock bars |
| IBKR Live | Verified live account in a separate local TWS session | Same account and data checks, read-only |

The default simulator uses $5,000 of virtual cash, whole shares, a $1,000
per-stock entry cap, next-bar fills, 43 bps per side and a 2% drawdown stop.
Its fixed trade cadence exercises the workflow; it is not a trading signal
or evidence of an edge. An optional `--runtime none` starts with no engine.
The earlier synthetic test engine is still available via `--runtime synthetic-smoke`.

## Launch

Install Python dependencies from the repository root. Add the optional TWS
extra if you intend to use the broker views:

```sh
python -m pip install -e ".[dev,tws]"
python scripts/launch_operator_console.py
```

The desktop launcher starts the loopback operator, generates an in-memory
operator session token if needed, checks paper-state health, and shuts down
the operator when Electron exits. Electron passes only allowlisted actions
through its preload bridge. The renderer has no network access or token.

For the browser, use the same local operator, with a fresh random token.
In PowerShell:

```powershell
$env:TRADING_OPERATOR_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:DISABLE_STUBS = "1"
python -m scripts.run_operator_api
```

Open `http://127.0.0.1:8765/workstation` and enter the operator token from
that local PowerShell session. Read it locally with `$env:TRADING_OPERATOR_TOKEN`
in the shell where you set it. The browser keeps it in the current tab's
memory. It is unrelated to your IBKR login. Use the managed launcher for the
desktop; do not serve the operator on a public or network address.

## Connect Trader Workstation

1. On the same computer as the operator, sign in to TWS in the intended
   environment. Enable **ActiveX and Socket Clients**, leave **Read-Only API**
   enabled, and confirm the port under **Global Configuration → API → Settings**.
2. Set `IBKR_TWS_PAPER_ACCOUNT` to the exact paper account ID shown by TWS
   or `IBKR_TWS_LIVE_ACCOUNT` to the exact live account ID. Restart the
   operator after changes. Do not enter an IBKR password in RomanBot.
3. Choose **TWS connection** and check the intended account. Use the regular
   account workspace to check another watchlist symbol.

| Setting | Paper default | Live default |
| --- | ---: | ---: |
| `IBKR_TWS_<MODE>_PORT` | 7497 | 7496 |
| `IBKR_TWS_<MODE>_CLIENT_ID` | 71 | 72 |

Ports can differ from defaults. When both profiles are configured, use
distinct ports and account IDs. A port or a successful socket connection is
not proof of account identity: the configured account must appear in TWS's
managed-account response. The UI shows only a masked ID, the installed
`ib_async` version, and the negotiated TWS API server version after a valid
check. The API server version is not the TWS application build; check TWS
**Help → About** for the installed build.

Account data is a timestamped snapshot. Each check connects on request and
disconnects afterward. Missing data entitlements may leave the chart empty
even when the account is verified. The chart uses five-minute regular-session
IBKR historical trade bars, not a streaming quote feed. IBKR account
currencies are shown as returned, and unknown values are left blank.

[IBKR TWS setup and port documentation](https://www.interactivebrokers.com/support/faq/content/api-connect-paper-trading-account-29950139)
· [IBKR API production release notes](https://www.ibkrguides.com/releasenotes/prod-2026.htm)
· [ib_async client package](https://pypi.org/project/ib_async/)

## Scope and verification

- The IBKR views do not submit or cancel broker orders and do not enable
  automatic strategies. Orders in the simulator are not IBKR positions.
  The earlier bracket examples and placeholder order router are not called.
- Research jobs use local CSVs and remain isolated from both TWS sessions.
- Run `npm run check`, `npm test` and `npm run dist` in
  `dashboard/electron-app`. The Desktop Operator workflow also checks and
  packages the Windows NSIS installer; the Windows installer needs a local
  operator backend. It does not install or upgrade TWS.
