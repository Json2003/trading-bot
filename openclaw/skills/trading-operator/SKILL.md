---
name: trading-operator
description: Safely observe and control the local paper-trading service without submitting arbitrary trades or resetting risk controls.
---

# Trading Operator

Use this skill only against the loopback Trading Bot Operator API.

## Required environment

- `TRADING_OPERATOR_URL` defaults to `http://127.0.0.1:8765`
- `TRADING_OPERATOR_TOKEN` contains the bearer token

Never print, log, summarize, or transmit the token.

## Allowed operations

- Read service status, open orders and positions.
- Start paper mode.
- Pause new strategy cycles.
- Stop the service and cancel open orders.
- Cancel all open orders.
- Trigger the emergency stop.

## Forbidden operations

- Do not submit an arbitrary order.
- Do not enable live trading.
- Do not change position size or financial risk limits.
- Do not reset a latched kill switch.
- Do not request or read broker credentials.
- Do not expose the operator API beyond the loopback interface.

## Requests

Set these shell variables before making requests:

```sh
BASE_URL="${TRADING_OPERATOR_URL:-http://127.0.0.1:8765}"
AUTH="Authorization: Bearer ${TRADING_OPERATOR_TOKEN}"
```

Read status:

```sh
curl --fail --silent --show-error -H "$AUTH" "$BASE_URL/operator/status"
```

Read orders:

```sh
curl --fail --silent --show-error -H "$AUTH" "$BASE_URL/operator/orders"
```

Read positions:

```sh
curl --fail --silent --show-error -H "$AUTH" "$BASE_URL/operator/positions"
```

Start paper mode:

```sh
curl --fail --silent --show-error -X POST -H "$AUTH" "$BASE_URL/operator/start-paper"
```

Pause:

```sh
curl --fail --silent --show-error -X POST -H "$AUTH" "$BASE_URL/operator/pause"
```

Stop and cancel open orders:

```sh
curl --fail --silent --show-error -X POST -H "$AUTH" "$BASE_URL/operator/stop"
```

Cancel all open orders:

```sh
curl --fail --silent --show-error -X POST -H "$AUTH" "$BASE_URL/operator/cancel-all"
```

Emergency stop:

```sh
curl --fail --silent --show-error -X POST -H "$AUTH" "$BASE_URL/operator/emergency-stop"
```

## Response behavior

Summarize the returned state clearly. When the kill switch is latched, state that manual recovery is required. Never attempt repeated start requests after a latched response.
