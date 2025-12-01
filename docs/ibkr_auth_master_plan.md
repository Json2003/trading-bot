# IBKR Web API Authentication Master Plan

This plan captures how to integrate Interactive Brokers' OAuth2 private_key_jwt authentication into this repository. It focuses on secure key handling, token acquisition, and request wiring so that the trading and account-management flows can call IBKR Web API endpoints safely.

## 1) Authentication approach
- Use OAuth 2.0 with **private_key_jwt** (RFC 7521/7523) instead of client secrets.
- The client authenticates by sending a signed `client_assertion` JWT to the IBKR authorization server; the server validates it against the registered public key.
- Access tokens are requested via `POST /api/v1/token` and then sent on every IBKR Web API request as `Authorization: Bearer <token>`.

## 2) Key management and registration
- Generate separate RSA keys for QA and production; required sizes are 3072 or 4096 bits in PEM format.
- Deliver the public key for each environment to IBKR and register the source IP CIDR ranges that will originate requests.
- Store private keys outside the repo (mounted secrets or env var file paths). Do **not** commit keys. Add config entries for:
  - `IBKR_AUTH_PRIVATE_KEY_PATH` (PEM file)
  - `IBKR_AUTH_KEY_ID` (kid claim used in JWT header)
  - `IBKR_AUTH_AUDIENCE` (token endpoint URL)
  - `IBKR_AUTH_CLIENT_ID`
- Capture a checklist to sign required service agreements before go-live.

## 3) Client assertion construction
- Build a helper (target location: `tradingbot_ibkr/services/ibkr_auth.py`) that:
  - Loads the RSA private key from `IBKR_AUTH_PRIVATE_KEY_PATH`.
  - Issues a JWT with claims `iss`, `sub`, `aud`, `jti`, `iat`, `exp` (~5 minutes), and header `kid`.
  - Signs with RS256. Avoid try/except around imports per repo style.
- Unit-test fixture: stub clock and key to produce deterministic JWT for snapshot tests.

## 4) Token request flow
- Implement `IBKRTokenClient` in `tradingbot_ibkr/services/ibkr_auth.py` with methods:
  - `request_token(scopes: list[str]) -> AccessToken`: POST to `/api/v1/token` with `grant_type=client_credentials`, `scope`, `client_assertion_type`, and signed `client_assertion`.
  - `refresh_if_needed()` to reuse unexpired tokens and pre-emptively refresh when ≤60s to expiry.
- Add telemetry hooks for rate-limit events and HTTP 429 handling; respect limits of **10 req/s per endpoint** and **600 req/min per master account**.
- Persist token metadata (expiry, scopes) in `model_store/logs/ibkr_tokens.json` for observability, without storing the raw JWT.

## 5) Request wiring
- Wrap IBKR HTTP calls behind an injected session in the execution layer (e.g., `tradingbot_ibkr/execution/ibkr_broker.py`).
- Add an auth middleware that attaches `Authorization: Bearer <token>` to every request and retries once on 401 with a forced refresh.
- Include structured errors for 429 (Too Many Requests) and surface retry-after guidance to the job runner.

## 6) Retail vs. institutional paths
- Retail/individual users can continue to run via the Client Portal Gateway, which performs local authentication; no private_key_jwt setup is needed.
- Institutional/third-party services must enable the RSA-based flow above. Document environment selection via a config flag `IBKR_AUTH_MODE` = `gateway|private_key_jwt`.

## 7) Acceptance criteria
- Key paths, client_id, audience, and kid are configurable without code changes.
- Token client produces valid assertions (signature verified in unit tests with public key) and successfully fetches tokens from mocked `/api/v1/token`.
- Live IBKR calls automatically refresh tokens and back off on 429s according to limits.
- Documentation lives in this file and is linked from onboarding/runbooks.

## 8) Next steps
- Add a short onboarding section to the main README referencing this plan.
- Implement `tradingbot_ibkr/services/ibkr_auth.py` per sections 3–5.
- Wire the token client into IBKR execution flows and add regression tests under `tradingbot_ibkr/tests/`.
