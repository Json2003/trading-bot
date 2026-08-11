# Safe Paper Deployment

This procedure deploys the local Electron operator with the synthetic paper runtime. It does not connect to an exchange, use API keys, or place live orders.

## Safety boundary

The deployment must remain:

- TRADING_OPERATOR_MODE=paper
- TRADING_OPERATOR_RUNTIME=synthetic-smoke or none
- bound to 127.0.0.1, localhost, or ::1
- credential-free
- protected by a locally generated operator token

Do not use configs/live.yaml or legacy server entry points for this deployment.

## Windows PowerShell setup

From the repository root:

~~~powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
$env:TRADING_OPERATOR_MODE = "paper"
$env:TRADING_OPERATOR_RUNTIME = "synthetic-smoke"
$env:TRADING_OPERATOR_HOST = "127.0.0.1"
$env:TRADING_OPERATOR_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
python scripts/validate_paper_deployment.py --require-token
~~~

## Start the paper API

In the same environment:

~~~powershell
python scripts/run_operator_api.py
~~~

The API must listen only on 127.0.0.1:8765. A missing token, non-loopback host, or live runtime must fail closed.

## Build and start the Electron operator

In a second PowerShell window:

~~~powershell
cd dashboard/electron-app
npm ci
npm run check
npm test
npm run dist
npm start
~~~

The Windows installer is a paper-testing artifact. It is not a signed production release and must not be distributed as a live-trading application.

## Manual acceptance checks

1. The API health endpoint reports mode: paper and kill_switch_latched: false.
2. The operator UI reports synthetic-smoke or no engine.
3. No exchange credentials are present in the process environment.
4. start-paper creates only paper-broker activity.
5. Emergency stop latches and remains latched after restart.
6. Research runs use only local CSV files and write only beneath var/paper_lab.
7. Closing the API cancels remaining paper orders and exits cleanly.

## Stop conditions

Stop immediately if the API binds to a non-loopback address, any exchange credential is requested, the UI reports live mode, or any order leaves the in-memory paper broker.
