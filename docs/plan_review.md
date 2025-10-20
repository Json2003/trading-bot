# Splitstar Development — Plan Review (WCOIN / Vibracoin / Vibra Payments)

Date: 2025-10-22
Source: Full project plan shared in chat (Split Star / Vibra / WCOIN (Remittix))

## Executive summary

The plan aims to launch a compliant, fiat‑redeemable, multi‑currency‑referenced stablecoin (WCOIN) with an SG issuer, consumer app (Vibracoin), and merchant rails (Vibra Payments). It is comprehensive across legal, regulatory, product/tech, and GTM. For this repository (trading-bot), the most relevant, near-term deliverables are: a management workspace (/manage), a reliable job runner and metrics, ingestion for FX/oracle data, and a PoR hash pipeline. Smart contracts, licensing, banking, and MAS workstreams will live outside this repo.

Top recommendations:

- Lock the chain choice early (Base vs Arbitrum vs Polygon PoS) to unblock contract scaffolding.
- Define public Basket Methodology v1.0 and Oracle Disclosure Policy; these drive contract params and off-chain jobs.
- Build the Issuer Console (admin) and the PoR daily hash pipeline in parallel; these are high-visibility compliance enablers.

## Workstreams and deliverables

### 1) Corporate & Governance

- File DE entities and SG issuer; execute IP/Brand/MSAs; set up multi‑currency bank/PSP.
- Board resolutions to adopt policies and authorize key accounts.
- Dependencies: Legal counsel (US/SG), bank partners.

### 2) Regulatory & Compliance (SG Issuer)

- Target MAS MPI or sandbox progression.
- Policy binder: AML/KYC, Treasury & Reserves, Oracle & Disclosure, Incident Response, Custody/Keys.
- Reporting: monthly attestation; event-driven disclosures; daily on‑chain PoR anchor.
- Dependencies: KYC vendor, txn monitoring, auditor.

### 3) Economics — Basket & Redeemability

- Publish methodology for fiat basket (Top‑20 turnover vs GDP vs equal weights with caps/floors).
- Reweighting cadence and notice periods (semi‑annual, 30‑day notice, 14‑day timelock).
- Operational daily rebalancing with ±0.5% drift bands.

### 4) Oracle & Index Methodology

- 3+ independent FX sources; median‑of‑means, anomaly rejection.
- Hourly baseline updates; fast path on large moves; circuit-breakers.
- On-chain signer rotation and calldata with Merkle commitment.
- Off-chain reproducible CSV + daily index, hashed to chain.

### 5) Chain & Contracts (EVM L2 recommended)

- Contracts: WCOIN ERC‑20 (EIP‑2612), IssuerController (role‑gated mint/burn), BasketIndex (weights & params, timelocked), OracleRelay, ProofOfReserves poster.
- Governance: 4/7 multisig; TimelockController; PAUSER role and clear break-glass.
- Monorepo structure for contracts/oracle/index SDKs and web apps.

### 6) Treasury & Reserves

- Segregated safeguarded client monies (SG), allowed instruments; daily rebalancer vs drift bands.
- Counterparties with multi‑currency capability and mapped settlement cutoffs.

### 7) Proof‑of‑Reserves (PoR)

- Daily: Merkle root of anonymized balances + types; post hash on-chain.
- Monthly: third‑party attestation; reproducible methodology.
- Incident disclosure for shortfalls or oracle faults.

### 8) Risk, Security & Keys

- HSM/MPC per role (Issuance, Oracle, Treasury, Admin).
- 2 external audits; invariant tests; monitoring for oracle divergence and supply anomalies.
- Runbooks for pause/unpause, rotations, failover, redemption surges.
- Bug bounty program.

### 9) Compliance Stack

- KYC/AML with tiers; sanctions/PEP; liveness.
- Transaction monitoring; TRAVEL rule (if required); SAR/STR workflows.

### 10) Apps & APIs

- Vibracoin consumer app: on/off‑ramp, wallet, swap/send, redeem, statements.
- Vibra Payments: REST + webhooks; checkout widget; pricing in local CCY; auto‑FX via oracle.

### 11) Brand & Collateral

- Logos complete; pending brand kit, website, docs/pitch deck.

### 12) Launch Plan (phases)

- Week 1: file entities, chain choice, monorepo init, draft policies.
- Month 1: oracle MVP on testnet; contracts on testnet; issuer console alpha; MAS prep; PoR pipeline on testnet.
- Quarter 1: external audit #1; close beta merchants; mainnet launch gated on MAS and audit.

### 13) Metrics & SLOs

- Peg deviation ≤ 30 bps (95% daily), redemption SLA ≤ 24h.
- Oracle divergence alarms < 1/week; uptime ≥ 99.9%.
- Reserves ≥ 100% at all times.

### 14) Open Decisions

- Chain selection; KYC vendor; bank/PSP; auditor/attestor.

### 15) RACI

- CEO (Jason), CFO (Mayra), COO (Caleb), Advisor (Leah), Splitstar Dev.

## Alignment with this repository (trading-bot)

This repo is a trading/backtesting/ops toolkit with a FastAPI server and dashboard stubs. It can power several parts of the plan (off-chain and operational services), while smart contracts and consumer/merchant apps belong in a separate monorepo.

What to build here (immediately actionable):

1) Management workspace `/manage` (issuer/operator console)
   - Toggles: PAPER, CONTINUOUS_BACKTEST, STRATEGY, RISK, EXCHANGE
   - Status: feeds latency/recency, account balances (if connected), job health, model status
   - Charts: equity/drawdown, live OHLCV + signals (for monitoring)
   - WS live updates; Settings via `GET/POST /settings`
2) Continuous evaluation loop (safe)
   - Interval checks (30–60s), incremental updates, skip if no new bars
   - Promotion gates: `ALLOW_MODEL_PROMOTE=true` + `allow_live_confirm.txt`
3) Oracle/FX ingestion MVP
   - Pull 3 FX sources (e.g., provider APIs or CCXT proxies where applicable)
   - Compute median‑of‑means, variance checks, circuit-breaker flags
   - Persist daily CSV snapshot + hash (for PoR linkage)
4) PoR daily hash pipeline (scaffold)
   - CLI/cron: accept anonymized balances JSON -> compute Merkle root -> store artifact -> (optional) post hash to chain via separate relayer service
   - Artifacts saved under `tradingbot_ibkr/model_store/logs/por/` with date partitions
5) Monitoring & metrics
   - `/metrics` JSON: data recency lag, loop p95, API error/rate-limit counts
   - Persist daily summary reports under `model_store/logs/`

What belongs in a new monorepo (per plan §5):

- Smart contracts (`wcoin-contracts`), oracle poster, index weighting, SDK, consumer app, issuer console (admin web), merchant app.

## Gap analysis vs current repo

- Missing: `/manage` page and `/settings` endpoints (easy to add).
- Missing: FX/oracle ingestion utilities (add simple adapters and median‑of‑means combiner).
- Missing: PoR hash job (Merkle tree util + job harness).
- Present: job runner, FastAPI server, backtest framework, dashboard stub, logging, optimization tooling.

## Proposed milestones (repo)

- M1 (1 week): `/settings` API + `/manage` page (toggles + live health); metrics endpoint; basic WS updates.
- M2 (2–3 weeks): FX ingestion MVP (3 sources) + median‑of‑means + CSV daily; PoR hash job scaffold; daily scheduler; ops docs.
- M3 (4–6 weeks): Hardening: retries/backoff, circuit-breakers; dashboards for oracle variance; PoR artifacts browser; CI tests.

Acceptance criteria examples:

- `/settings` persists toggles; UI reflects changes without reload; changes logged.
- Metrics show recency lag < 90s p95; loop p95 < 2s under nominal load.
- FX combiner flags divergent sources and halts downstream when threshold exceeded; daily CSV and SHA256 are written.
- PoR job produces deterministic Merkle root given the same input; artifacts reproducible.

## Risks and mitigations

- Geo-blocked exchange APIs → Use provider mix, region placement, or proxies.
- Compliance timelines → Start policy drafts and vendor outreach in parallel with tech.
- Oracle integrity → Multi-source, anomaly rejection, signed updates, and auditable CSVs.
- Key management → Adopt HSM/MPC early and limit hot keys; timelock admin actions.

## Immediate backlog (this repo)

- Backend
  - Add `/settings` (GET/POST) in `server.py`; persist to `STATE` and optionally `.env` overlay
  - Add `/metrics` exposing loop durations, recency, error counts
  - WS topic `status` broadcasting health snapshots
- Frontend
  - Create `/manage` template/page in `tradingbot_ibkr/dashboard.py` or as a simple SPA served by FastAPI
  - Cards: Feeds, Accounts, Trades, Models; Charts: Equity/Drawdown, Live OHLCV
- Jobs
  - `scripts/fx_collect.py` (fetch 3 sources) and `scripts/fx_daily_snapshot.py` (write CSV+hash)
  - `scripts/por_hash_job.py` (Merkle root from balances JSON; write artifacts)
- Docs & Tests
  - Update `INGESTION_AND_OPS.md`; add README sections for `/manage`, settings, metrics
  - Unit tests for settings persistence and PoR hash determinism

## Links

- Splitstar plan (external path): C:\\Users\\j-mga\\OneDrive\\Documents\\GitHub\\Wcoin
- Repo-aligned summary: docs/splitstar_plan.md
- Printable checklist: docs/copilot_checklist.md
