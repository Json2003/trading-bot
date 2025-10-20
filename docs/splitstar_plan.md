# Splitstar Development — Roadmap Alignment

This repository aligns execution with the external Splitstar Development plan.

Plan source (external):

- Windows path (owner-provided): `C:\Users\j-mga\OneDrive\Documents\GitHub\Wcoin`
- If a public link is available, add it below.

Related documents in this repo:

- Plan review summary: `docs/plan_review.md`

How to keep this repo in sync

- Export key milestones and IDs from the Splitstar Development plan into this file.
- Reference the milestone IDs in PR titles/descriptions (e.g., `[SPLITSTAR-12] Manage page toggles`).
- Update the checklist in `docs/copilot_checklist.md` when priorities change.

Initial priorities to reflect in this codebase

- Management workspace `/manage` with:
  - Feature toggles (PAPER, CONTINUOUS_BACKTEST, STRATEGY, RISK, EXCHANGE)
  - Live status for feeds, accounts, trades, models (via FastAPI + WebSocket)
  - Charts: equity curve, drawdown, live OHLCV with signals
- Continuous evaluation loop (interval-based) with safe promotion gates
- Ingestion health and data freshness metrics

Owner actions

- Provide a shareable link or export of the Splitstar Development plan (optional but recommended).
- Confirm canonical KPIs and naming (latency, recency, winrate, Sharpe, drawdown).

Repo actions

- Keep `/docs/splitstar_plan.md` current with milestones.
- Link this file from `.github/copilot-instructions.md` and `docs/copilot_checklist.md`.
