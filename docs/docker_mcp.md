# Docker & MCP Deployment Guide

This guide covers running the trading bot inside Docker and wiring it to an MCP (Model Control Platform) server.

## 1. Prepare environment variables

Copy the template to `env/docker.env` and adjust values:

```bash
cp env/docker.env.example env/docker.env
```

Key variables:

- `SECRET_KEY`: JWT secret for API auth.
- `MCP_BASE_URL`: Base URL to your MCP server (leave blank to disable integration).
- `MCP_API_KEY`: Optional bearer token for MCP.
- `EXCHANGE` / `PAPER`: Strategy runtime configuration.

## 2. Build the container

From the repo root:

```bash
docker build -f docker/Dockerfile -t trading-bot:latest .
```

## 3. Run with docker-compose

```bash
docker compose -f docker/docker-compose.yml --project-directory . up --build
```

This exposes the FastAPI service at `http://localhost:8000`. MCP endpoints become available when `MCP_BASE_URL` is set:

- `GET /mcp/health`
- `GET /mcp/signals`
- `POST /mcp/metrics`

## 4. Verify MCP connectivity

```bash
curl http://localhost:8000/mcp/health
```

If MCP is disabled or misconfigured, the API returns `404` or `502`. Check container logs for detailed errors.

## 5. Notes

- Mounting `data/` and `model_store/` keeps state between container runs (configured in `docker-compose.yml`).
- To run without compose, pass env vars manually:

  ```bash
  docker run --rm -p 8000:8000 \
    -e SECRET_KEY=change-this \
    -e MCP_BASE_URL=https://mcp.example.com/api \
    -e MCP_API_KEY=token \
    trading-bot:latest
  ```

- Disable MCP integration by omitting `MCP_BASE_URL`.
