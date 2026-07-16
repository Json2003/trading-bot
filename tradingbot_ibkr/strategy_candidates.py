"""Immutable strategy candidate packages and explicit live-readiness gates.

Research winners can be registered here for paper probation. The registry never
places orders or activates live trading. It only emits a runtime configuration
after all quantitative, operational and human-approval gates are satisfied.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    source_job_id: str
    source_account_id: str
    dataset_id: str
    total_return: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    trade_count: int
    score: float


@dataclass(frozen=True, slots=True)
class PaperProbationEvidence:
    trading_days: int
    trade_count: int
    total_return: float
    max_drawdown: float
    profit_factor: float
    reconciliation_passed: bool
    restart_recovery_passed: bool
    duplicate_order_test_passed: bool
    kill_switch_test_passed: bool


@dataclass(slots=True)
class StrategyCandidate:
    candidate_id: str
    created_at: str
    strategy: str
    execution_policy: str
    params: dict[str, float]
    risk: dict[str, float]
    research: CandidateEvidence
    fingerprint: str
    stage: str = "research_candidate"
    paper_probation: PaperProbationEvidence | None = None
    approval_recorded_at: str | None = None
    approval_note: str | None = None
    live_eligible: bool = False
    gate_results: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyCandidateRegistry:
    """Persist reproducible candidates and evaluate live-export eligibility."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._candidates: dict[str, StrategyCandidate] = {}
        self._load()

    def register(
        self,
        *,
        strategy: str,
        execution_policy: str,
        params: Mapping[str, float],
        risk: Mapping[str, float],
        research: CandidateEvidence,
    ) -> dict[str, Any]:
        strategy_name = str(strategy).strip()
        policy_name = str(execution_policy).strip()
        if not strategy_name or not policy_name:
            raise ValueError("strategy and execution_policy are required")

        normalized_params = {str(key): float(value) for key, value in params.items()}
        normalized_risk = {str(key): float(value) for key, value in risk.items()}
        payload = {
            "strategy": strategy_name,
            "execution_policy": policy_name,
            "params": normalized_params,
            "risk": normalized_risk,
            "research": asdict(research),
        }
        fingerprint = _fingerprint(payload)

        with self._lock:
            for candidate in self._candidates.values():
                if candidate.fingerprint == fingerprint:
                    return candidate.as_dict()
            candidate = StrategyCandidate(
                candidate_id=uuid.uuid4().hex,
                created_at=_utc_now(),
                strategy=strategy_name,
                execution_policy=policy_name,
                params=normalized_params,
                risk=normalized_risk,
                research=research,
                fingerprint=fingerprint,
            )
            candidate.gate_results = self._evaluate(candidate)
            self._candidates[candidate.candidate_id] = candidate
            self._save()
            return candidate.as_dict()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._candidates.values(), key=lambda item: item.created_at, reverse=True)
            return [item.as_dict() for item in ordered]

    def get(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            return self._require(candidate_id).as_dict()

    def record_paper_probation(
        self,
        candidate_id: str,
        evidence: PaperProbationEvidence,
    ) -> dict[str, Any]:
        with self._lock:
            candidate = self._require(candidate_id)
            candidate.paper_probation = evidence
            candidate.stage = "paper_probation_complete"
            candidate.live_eligible = False
            candidate.gate_results = self._evaluate(candidate)
            self._save()
            return candidate.as_dict()

    def record_human_approval(self, candidate_id: str, *, approval_note: str) -> dict[str, Any]:
        note = str(approval_note).strip()
        if len(note) < 12:
            raise ValueError("approval_note must document the approval decision")
        with self._lock:
            candidate = self._require(candidate_id)
            candidate.approval_recorded_at = _utc_now()
            candidate.approval_note = note
            candidate.gate_results = self._evaluate(candidate)
            candidate.live_eligible = all(candidate.gate_results.values())
            candidate.stage = "live_eligible" if candidate.live_eligible else "gates_incomplete"
            self._save()
            return candidate.as_dict()

    def export_runtime_config(self, candidate_id: str) -> dict[str, Any]:
        """Return an immutable live-capable config only after every gate passes."""

        with self._lock:
            candidate = self._require(candidate_id)
            candidate.gate_results = self._evaluate(candidate)
            candidate.live_eligible = all(candidate.gate_results.values())
            if not candidate.live_eligible:
                missing = sorted(name for name, passed in candidate.gate_results.items() if not passed)
                raise RuntimeError(f"candidate is not live eligible; incomplete gates: {missing}")
            config = {
                "candidate_id": candidate.candidate_id,
                "fingerprint": candidate.fingerprint,
                "strategy": candidate.strategy,
                "execution_policy": candidate.execution_policy,
                "params": dict(candidate.params),
                "risk": dict(candidate.risk),
                "approval_recorded_at": candidate.approval_recorded_at,
            }
            config["export_fingerprint"] = _fingerprint(config)
            return config

    def _evaluate(self, candidate: StrategyCandidate) -> dict[str, bool]:
        research = candidate.research
        paper = candidate.paper_probation
        return {
            "research_positive_return": research.total_return > 0,
            "research_sharpe": research.sharpe >= 1.0,
            "research_profit_factor": research.profit_factor >= 1.20,
            "research_trade_sample": research.trade_count >= 30,
            "research_drawdown": research.max_drawdown <= 0.15,
            "paper_probation_present": paper is not None,
            "paper_days": bool(paper and paper.trading_days >= 20),
            "paper_trade_sample": bool(paper and paper.trade_count >= 50),
            "paper_positive_return": bool(paper and paper.total_return > 0),
            "paper_profit_factor": bool(paper and paper.profit_factor >= 1.15),
            "paper_drawdown": bool(paper and paper.max_drawdown <= 0.10),
            "broker_reconciliation": bool(paper and paper.reconciliation_passed),
            "restart_recovery": bool(paper and paper.restart_recovery_passed),
            "duplicate_order_protection": bool(paper and paper.duplicate_order_test_passed),
            "kill_switch": bool(paper and paper.kill_switch_test_passed),
            "human_approval": candidate.approval_recorded_at is not None,
        }

    def _require(self, candidate_id: str) -> StrategyCandidate:
        candidate = self._candidates.get(str(candidate_id))
        if candidate is None:
            raise KeyError(candidate_id)
        return candidate

    def _save(self) -> None:
        payload = [candidate.as_dict() for candidate in self._candidates.values()]
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in payload if isinstance(payload, list) else []:
            try:
                research = CandidateEvidence(**item["research"])
                paper_payload = item.get("paper_probation")
                paper = PaperProbationEvidence(**paper_payload) if paper_payload else None
                candidate = StrategyCandidate(
                    candidate_id=str(item["candidate_id"]),
                    created_at=str(item["created_at"]),
                    strategy=str(item["strategy"]),
                    execution_policy=str(item["execution_policy"]),
                    params={str(key): float(value) for key, value in item.get("params", {}).items()},
                    risk={str(key): float(value) for key, value in item.get("risk", {}).items()},
                    research=research,
                    fingerprint=str(item["fingerprint"]),
                    stage=str(item.get("stage", "research_candidate")),
                    paper_probation=paper,
                    approval_recorded_at=item.get("approval_recorded_at"),
                    approval_note=item.get("approval_note"),
                    live_eligible=bool(item.get("live_eligible", False)),
                    gate_results={
                        str(key): bool(value) for key, value in item.get("gate_results", {}).items()
                    },
                )
                candidate.gate_results = self._evaluate(candidate)
                candidate.live_eligible = all(candidate.gate_results.values())
                self._candidates[candidate.candidate_id] = candidate
            except (KeyError, TypeError, ValueError):
                continue


__all__ = [
    "CandidateEvidence",
    "PaperProbationEvidence",
    "StrategyCandidate",
    "StrategyCandidateRegistry",
]
