# Security guidance for trading-bot

This repository includes a demo server and research tooling. The defaults are intentionally permissive for local development. Use the guidance below to harden a production deployment.

## Authentication and authorization

- Do not use the mock token flow in production. The demo `server.py` accepts a simplified token for local testing.
  - Enable real JWTs using `PyJWT` (or your IdP) and configure a strong `JWT_SECRET` (256-bit) and `JWT_ALG` (e.g., `HS256` or `RS256`).
  - Require HTTPS end-to-end and validate token expiration (`exp`), issuer (`iss`), audience (`aud`).
  - In production set environment variables:
    - `ENV=prod`
    - `MOCK_AUTH=false` (or remove any dev-only flags)
    - `JWT_SECRET` and `JWT_ALG`
- Prefer short-lived access tokens with refresh flow managed outside this service.

## CORS and rate limiting

- In development CORS may be open; in production restrict to explicit origins via `ALLOWED_ORIGINS` (comma-separated).
- Keep basic rate limiting enabled server-side. Tune limits to expected QPS and enable per-IP and per-auth-subject buckets.

## Secrets and configuration

- Never commit secrets. Provide values via environment variables, container secrets, or your cloud secret manager.
- For local dev, use a `.env` file that is excluded by `.gitignore`.
- For CI/CD, store secrets as encrypted repository or environment secrets.

## Data and filesystem access

- The app writes to `tradingbot_ibkr/datafiles/` and `tradingbot_ibkr/model_store/`.
  - Run with a non-root user and least-privilege filesystem permissions.
  - When using cloud storage (e.g., GCS), scope keys to read/write only the required buckets/prefixes. Rotate regularly.
- Treat generated model artifacts as sensitive. Store with access controls and integrity checksums where possible.

## Dependency and supply chain hygiene

- Use pinned dependencies in `requirements.txt` and `tradingbot_ibkr/requirements.txt`.
- Regularly audit dependencies with tools like `pip-audit` and renovate/dependabot.
- Build reproducibly (lockfiles, hashes) and verify images with SBOM and signing where applicable.

## Logging and observability

- Avoid logging PII, tokens, or secrets. Redact sensitive fields at the boundary.
- Set log level to `INFO` or lower-verbosity in production; capture structured logs and ship to a central store.

## Network and process security

- Run behind a TLS-terminating reverse proxy or API gateway. Enforce HSTS and modern TLS ciphers.
- Consider a Web Application Firewall (WAF) and basic anomaly detection for request patterns.
- Containerize with a read-only root filesystem where possible; drop capabilities and set seccomp/apparmor profiles.

## Promotion safeguards for live trading

- This repo requires two conditions before promoting a model to live trading:
  - `ALLOW_MODEL_PROMOTE=true` in the environment, and
  - A file `allow_live_confirm.txt` at the repo root.
- Keep both disabled by default and require a manual, authenticated change for promotion.

## Reporting vulnerabilities

If you believe you have found a security issue, please open a private security advisory or contact the maintainers out-of-band. Do not file public issues with exploit details.
