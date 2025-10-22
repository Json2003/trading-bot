# Splitstar Operations Console
Splitstar Operations Console (formerly the trading-bot project) is the
operations and research surface for Splitstar's agent-based trading stack.
It keeps the `tradingbot_ibkr` package for backwards-compatible scripts while
rebranding the product experience, dashboards, and deployment targets under the
Splitstar Operations umbrella.

> 💡 **Repository rename:** clone the repo into `splitstar-operations-console`
> (or update existing checkouts) to match the new product branding. Scripts
> continue to recognise the legacy `trading-bot` directory for compatibility.

## Desktop installation

Run one of the provided scripts to set up a virtual environment and install
dependencies.

On Linux or macOS:

```bash
./install.sh
```

On Windows (cmd or PowerShell):

```bat
install.bat
```

This creates a `venv` directory and installs packages from
`tradingbot_ibkr/requirements.txt`.

## Docker & MCP deployment

- Copy `env/docker.env.example` to `env/docker.env` and set MCP/API credentials.
- Build the container and run via docker compose:

  ```bash
  docker compose -f docker/docker-compose.yml --project-directory . up --build
  ```

- MCP endpoints (`/mcp/health`, `/mcp/signals`, `/mcp/metrics`) are exposed automatically when `MCP_BASE_URL` is set.
  See `docs/docker_mcp.md` for details.

## Asset classes

The console supports multiple asset classes including forex, options, futures,
crypto, and stocks via a unified `AssetClass` enum. Trading scripts and the
engine can adjust risk settings based on the selected class.

## Binance to GCS ingestion

Fetch minute and five-minute klines for BTCUSDT and ETHUSDT across spot and
USDT-margined futures markets and upload them to a Google Cloud Storage bucket:

```bash
python tradingbot_ibkr/binance_to_gcs.py --bucket <bucket-name> \
  --symbols BTCUSDT,ETHUSDT --intervals 1m,5m --markets spot,um \
  --start 2024-01-01T00:00:00 --end 2024-01-01T01:00:00
```

Replace the bucket name and time range as needed.

## LangChain agent deployment

- Treat your LangSmith API token (for example `LANGSMITH_API_KEY=<your-token>`) as a
  secret. Export it at runtime or load it from a secrets manager, and never check the
  real value into source control.
- Install the latest LangChain SDK packages:

  ```bash
  pip install -U langchain langgraph langchainhub
  ```

- When you want LangSmith dashboards, set tracing environment variables before
  launching the app:

  ```bash
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_ENDPOINT=https://api.langchain.plus
  export LANGCHAIN_PROJECT=<project-name>
  ```

- Build agents with `langchain.agents` or LangGraph primitives. Inject the API
  key through LangChain settings (`langchain.settings.set_headers`) or by
  supplying `headers={"x-api-key": os.environ["LANGCHAIN_API_KEY"]}` when calling
  hosted endpoints.
- To hit LangChain's hosted Agents API, create a client:

  ```python
  from langchain.clients import AgentsClient
  client = AgentsClient(base_url=<your-endpoint>, api_key=os.environ["LANGCHAIN_API_KEY"])
  ```

  Each request must include the bearer token (Authorization header) and any
  deployment-specific headers.
- Forward run telemetry to LangSmith so you can inspect traces, tool usage, and
  latency in production.
- Harden operations by rotating the token regularly, limiting its scope, and
  keeping detailed audit logs for every agent action—especially before executing
  trades against live capital.
- The FastAPI gateway now exposes `/agents/run` (requires authentication with
  `write` permission) to forward payloads to a hosted LangChain agent using the
  configured credentials.
