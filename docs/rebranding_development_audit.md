# Splitstar Operations Console — Rebranding & Development Audit

## Scope and limitations
- This audit reviews the `trading-bot` repository because other `json2003` projects are not accessible from the current workspace.
- Findings focus on rebranding tasks required to align assets with the **Splitstar Operations Console** name and on development priorities already captured in the Splitstar roadmap.

## Rebranding priorities (rename "trading bot" → "Splitstar Operations Console")
| Area | Current state | Required action |
| --- | --- | --- |
| Top-level README | Header and tagline still read "trading bot / tradingbot", providing no Splitstar framing.【F:README.md†L1-L37】 | Update title, summary, and feature overview to introduce the Splitstar Operations Console and reference its role within the Splitstar program. |
| Setup guide | `SETUP_GUIDE.md` markets the app as an "Enhanced Trading Bot" and repeats the legacy name throughout the installation steps.【F:SETUP_GUIDE.md†L1-L30】【F:SETUP_GUIDE.md†L62-L158】 | Rewrite headings, descriptions, and examples to use the Splitstar Operations Console branding, including repo folder names if they change. |
| Ops/ingestion manual | `INGESTION_AND_OPS.md` titles the document "Ingestion & Ops — trading-bot" and instructs cloning into a `trading-bot` directory.【F:INGESTION_AND_OPS.md†L1-L33】 | Align the title, clone instructions, and narrative with the Splitstar Operations Console brand and any new repository slug. |
| Docker/MCP docs | The Docker deployment guide describes building an image tagged `trading-bot:latest` and refers to "the trading bot" in prose.【F:docs/docker_mcp.md†L1-L59】 | Retag images, service names, and descriptive text to match the new product name and update compose/project identifiers. |
| Developer instructions | `.github/copilot-instructions.md` labels the project "trading-bot" and enumerates modules under that name.【F:.github/copilot-instructions.md†L1-L70】 | Refresh internal guidance so automated agents and contributors use the Splitstar Operations Console vocabulary. |
| Additional assets | Numerous scripts, tests, and helper files describe themselves as part of "the trading bot" (e.g., automation scripts, dashboards).【F:docs/copilot_checklist.md†L1-L125】【F:dashboard/README.md†L3-L7】 | Sweep remaining files (scripts, dashboards, comments, test fixtures) to ensure copy, log prefixes, and package names reflect the Splitstar brand; decide whether to rename the `tradingbot_ibkr` package for full alignment. |

## Development priorities (from Splitstar roadmap)
- Implement the `/manage` operator workspace with live toggles, status cards, and monitoring charts to serve Splitstar issuer operations.【F:docs/plan_review.md†L102-L167】
- Add `/settings` and `/metrics` APIs plus WebSocket health broadcasting to back the console UI and automation hooks.【F:docs/plan_review.md†L131-L167】
- Build the FX/oracle ingestion MVP (three data sources, median-of-means aggregation, daily CSV + hash) to feed Splitstar treasury and oracle reporting workflows.【F:docs/plan_review.md†L116-L149】
- Scaffold the daily Proof-of-Reserves hash job and artifact storage to enable regulatory disclosures.【F:docs/plan_review.md†L120-L149】
- Harden monitoring, scheduling, and CI as outlined in the M1–M3 milestones so the console can graduate from prototype to production service.【F:docs/plan_review.md†L138-L173】

## Recommended next steps
1. Confirm the final product name, repository slug, and package prefix with Splitstar stakeholders so code renames can be scripted safely.
2. Execute the documentation sweep above, then update runtime identifiers (environment variables, Docker tags, FastAPI metadata, dashboard headers) in a follow-up change.
3. Prioritize the roadmap-aligned development backlog to deliver operator-facing functionality alongside the rebrand.
