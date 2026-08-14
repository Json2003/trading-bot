"""Bounded automation around the isolated paper strategy tournament.

The service explores only allowlisted strategy families and execution policies.
It uses development data for search, a separate selection holdout to choose one
locked finalist, and a later confirmation holdout for the reported result. The
confirmation result is not used to search across finalists. No result can
modify active trading configuration or risk limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from threading import Event, RLock, Thread
from typing import Any, cast
import uuid

import pandas as pd

from .strategy_candidates import CandidateEvidence, StrategyCandidateRegistry
from .paper_lab import (
    EXECUTION_POLICIES,
    STRATEGY_FAMILIES,
    ExecutionAssumptions,
    PaperStrategyTournament,
    StrategyProfile,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class LabRunSpec:
    dataset_id: str
    generations: int = 3
    accounts_per_generation: int = 12
    final_holdout_fraction: float = 0.25
    seed: int = 7


@dataclass(slots=True)
class LabJob:
    job_id: str
    spec: LabRunSpec
    state: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    generation: int = 0
    total_generations: int = 0
    candidates_evaluated: int = 0
    development_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    selection_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    final_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    locked_account_id: str | None = None
    error: str | None = None
    cancellation_requested: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["promotion_allowed"] = False
        payload["manual_review_required"] = True
        return _json_safe(payload)


class PaperLabAutomationService:
    """Run one bounded local strategy-learning job at a time."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        artifact_root: Path,
        max_generations: int = 6,
        max_accounts_per_generation: int = 24,
    ) -> None:
        self._dataset_root = Path(dataset_root).resolve()
        self._artifact_root = Path(artifact_root).resolve()
        self._jobs_dir = self._artifact_root / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._max_generations = max(1, int(max_generations))
        self._max_accounts = max(4, int(max_accounts_per_generation))
        self._jobs: dict[str, LabJob] = {}
        self._threads: dict[str, Thread] = {}
        self._cancel_events: dict[str, Event] = {}
        self._lock = RLock()
        self._load_jobs()

    def datasets(self) -> list[dict[str, Any]]:
        if not self._dataset_root.exists():
            return []
        results: list[dict[str, Any]] = []
        for path in sorted(self._dataset_root.rglob("*.csv")):
            if not path.is_file():
                continue
            results.append(
                {
                    "dataset_id": path.relative_to(self._dataset_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return results

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [job.as_dict() for job in ordered]

    def job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.as_dict()

    def start(self, spec: LabRunSpec) -> dict[str, Any]:
        validated = self._validate_spec(spec)
        dataset_path = self._resolve_dataset(validated.dataset_id)
        with self._lock:
            if any(job.state in {"queued", "running"} for job in self._jobs.values()):
                raise RuntimeError("a paper lab job is already active")
            job = LabJob(
                job_id=uuid.uuid4().hex,
                spec=validated,
                total_generations=validated.generations,
            )
            cancel_event = Event()
            thread = Thread(
                target=self._run_job,
                args=(job.job_id, dataset_path, cancel_event),
                name=f"paper-lab-{job.job_id[:8]}",
                daemon=True,
            )
            self._jobs[job.job_id] = job
            self._cancel_events[job.job_id] = cancel_event
            self._threads[job.job_id] = thread
            self._save_job(job)
        thread.start()
        return job.as_dict()

    def stage_finalist(
        self,
        job_id: str,
        account_id: str,
        registry: StrategyCandidateRegistry,
    ) -> dict[str, Any]:
        """Stage a completed held-out finalist for explicit later review."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state != "completed":
                raise ValueError("only completed research jobs can stage finalists")
            finalist = next(
                (item for item in job.final_leaderboard if item.get("account_id") == account_id),
                None,
            )
            if finalist is None:
                raise KeyError(account_id)
            if job.locked_account_id != account_id:
                raise ValueError(
                    "only the preselected finalist may be staged; the confirmation "
                    "holdout is locked and cannot be searched"
                )
            params = finalist.get("params")
            risk = finalist.get("risk")
            if not isinstance(params, dict) or not isinstance(risk, dict):
                raise ValueError("finalist package is missing parameters or risk configuration")
            total_return = _finite_metric(finalist, "total_return")
            max_drawdown = _finite_metric(finalist, "max_drawdown")
            sharpe = _finite_metric(finalist, "sharpe")
            profit_factor = _finite_metric(finalist, "profit_factor")
            score = _finite_metric(finalist, "score")
            trade_count = int(finalist.get("trade_count", 0))
            if trade_count < 1:
                raise ValueError("locked finalist has no finite trade sample")
            evidence = CandidateEvidence(
                source_job_id=job.job_id,
                source_account_id=str(finalist["account_id"]),
                dataset_id=job.spec.dataset_id,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe=sharpe,
                profit_factor=profit_factor,
                trade_count=trade_count,
                score=score,
                execution_cost_bps=28.0,
                costs_included=True,
                holdout_passed=True,
                holdout_trade_count=trade_count,
                holdout_total_return=total_return,
                holdout_max_drawdown=max_drawdown,
                holdout_profit_factor=profit_factor,
            )
            return registry.register(
                strategy=str(finalist["strategy"]),
                execution_policy=str(finalist["execution_policy"]),
                params={str(key): float(value) for key, value in params.items()},
                risk={str(key): float(value) for key, value in risk.items()},
                research=evidence,
            )

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state not in {"queued", "running"}:
                return job.as_dict()
            job.cancellation_requested = True
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
            self._save_job(job)
            return job.as_dict()

    def close(self) -> None:
        with self._lock:
            active = [
                (job_id, thread)
                for job_id, thread in self._threads.items()
                if thread.is_alive()
            ]
            for job_id, _ in active:
                event = self._cancel_events.get(job_id)
                if event is not None:
                    event.set()
        for _, thread in active:
            thread.join(timeout=5)

    def _validate_spec(self, spec: LabRunSpec) -> LabRunSpec:
        dataset_id = str(spec.dataset_id).strip()
        if not dataset_id:
            raise ValueError("dataset_id is required")
        generations = int(spec.generations)
        accounts = int(spec.accounts_per_generation)
        holdout = float(spec.final_holdout_fraction)
        if not 1 <= generations <= self._max_generations:
            raise ValueError(f"generations must be between 1 and {self._max_generations}")
        if not 4 <= accounts <= self._max_accounts:
            raise ValueError(f"accounts_per_generation must be between 4 and {self._max_accounts}")
        if not 0.20 <= holdout <= 0.40:
            raise ValueError("final_holdout_fraction must be between 20% and 40%")
        return LabRunSpec(
            dataset_id=dataset_id,
            generations=generations,
            accounts_per_generation=accounts,
            final_holdout_fraction=holdout,
            seed=int(spec.seed),
        )

    def _resolve_dataset(self, dataset_id: str) -> Path:
        candidate = (self._dataset_root / dataset_id).resolve()
        try:
            candidate.relative_to(self._dataset_root)
        except ValueError as exc:
            raise ValueError("dataset must be inside the configured dataset directory") from exc
        if candidate.suffix.lower() != ".csv" or not candidate.is_file():
            raise ValueError("dataset_id must reference an existing CSV dataset")
        return candidate

    def _run_job(self, job_id: str, dataset_path: Path, cancel_event: Event) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = "running"
            job.started_at = _utc_now()
            self._save_job(job)

        try:
            frame = pd.read_csv(dataset_path)
            spec = job.spec
            final_rows = max(40, int(len(frame) * spec.final_holdout_fraction))
            minimum_rows = max(160, 2 * final_rows + 80)
            if len(frame) < minimum_rows:
                raise ValueError(
                    "learning jobs require enough rows for development, selection, "
                    f"and confirmation holdouts (at least {minimum_rows})"
                )
            development_rows = len(frame) - 2 * final_rows
            development = cast(
                pd.DataFrame,
                frame.iloc[:development_rows].reset_index(drop=True),
            )
            selection_frame = frame.iloc[
                development_rows : development_rows + final_rows
            ].reset_index(drop=True)
            confirmation_start = development_rows + final_rows
            confirmation_frame = frame.iloc[confirmation_start:].reset_index(drop=True)
            warmup_rows = min(max(60, final_rows), len(development))
            selection_data: pd.DataFrame = pd.concat(
                [development.iloc[-warmup_rows:], selection_frame],
                ignore_index=True,
            )
            selection_fraction = final_rows / len(selection_data)
            confirmation_warmup_start = max(0, confirmation_start - warmup_rows)
            confirmation_data: pd.DataFrame = pd.concat(
                [
                    frame.iloc[confirmation_warmup_start:confirmation_start],
                    confirmation_frame,
                ],
                ignore_index=True,
            )
            confirmation_fraction = final_rows / len(confirmation_data)
            rng = random.Random(spec.seed)
            assumptions = ExecutionAssumptions()
            elites: list[StrategyProfile] = []

            for generation in range(1, spec.generations + 1):
                if cancel_event.is_set():
                    self._mark_cancelled(job_id)
                    return
                profiles = _build_generation(
                    rng,
                    generation=generation,
                    count=spec.accounts_per_generation,
                    elites=elites,
                )
                report = PaperStrategyTournament(
                    profiles,
                    assumptions=assumptions,
                    holdout_fraction=0.30,
                ).run(development)
                profile_map = {profile.account_id: profile for profile in profiles}
                elite_count = max(2, min(6, len(profiles) // 4))
                elites = [profile_map[item.account_id] for item in report.leaderboard[:elite_count]]
                summary = [_report_summary(item) for item in report.leaderboard]
                with self._lock:
                    job = self._jobs[job_id]
                    job.generation = generation
                    job.candidates_evaluated += len(profiles)
                    job.development_leaderboard = summary
                    self._save_job(job)

            if cancel_event.is_set():
                self._mark_cancelled(job_id)
                return

            finalists = _build_finalists(rng, elites, spec.accounts_per_generation)
            profile_map = {profile.account_id: profile for profile in finalists}
            selection_report = PaperStrategyTournament(
                finalists,
                assumptions=assumptions,
                holdout_fraction=selection_fraction,
            ).run(selection_data)
            selection_summary = [
                _report_summary(
                    item,
                    profile_map.get(item.account_id),
                    evaluation_role="selection_holdout",
                )
                for item in selection_report.leaderboard
            ]
            if not selection_report.leaderboard:
                raise ValueError("selection holdout produced no finalist report")
            selected_account_id = selection_report.leaderboard[0].account_id
            selected_profile = profile_map[selected_account_id]
            confirmation_report = PaperStrategyTournament(
                [selected_profile],
                assumptions=assumptions,
                holdout_fraction=confirmation_fraction,
            ).run(confirmation_data)
            with self._lock:
                job = self._jobs[job_id]
                job.selection_leaderboard = selection_summary
                job.locked_account_id = selected_account_id
                job.final_leaderboard = [
                    _report_summary(
                        item,
                        selected_profile,
                        evaluation_role="locked_confirmation_holdout",
                    )
                    for item in confirmation_report.leaderboard
                ]
                job.state = "completed"
                job.finished_at = _utc_now()
                self._save_job(job)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = _utc_now()
                self._save_job(job)

    def _mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = "cancelled"
            job.finished_at = _utc_now()
            self._save_job(job)

    def _save_job(self, job: LabJob) -> None:
        path = self._jobs_dir / f"{job.job_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job.as_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load_jobs(self) -> None:
        for path in self._jobs_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                spec = LabRunSpec(**payload["spec"])
                job = LabJob(
                    job_id=str(payload["job_id"]),
                    spec=spec,
                    state=str(payload.get("state", "failed")),
                    created_at=str(payload.get("created_at", _utc_now())),
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    generation=int(payload.get("generation", 0)),
                    total_generations=int(payload.get("total_generations", spec.generations)),
                    candidates_evaluated=int(payload.get("candidates_evaluated", 0)),
                    development_leaderboard=list(payload.get("development_leaderboard", [])),
                    selection_leaderboard=list(payload.get("selection_leaderboard", [])),
                    final_leaderboard=list(payload.get("final_leaderboard", [])),
                    locked_account_id=payload.get("locked_account_id"),
                    error=payload.get("error"),
                    cancellation_requested=bool(payload.get("cancellation_requested", False)),
                )
                if job.state in {"queued", "running"}:
                    job.state = "interrupted"
                    job.finished_at = _utc_now()
                self._jobs[job.job_id] = job
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue


def _random_params(rng: random.Random, strategy: str) -> dict[str, float]:
    if strategy == "ema_momentum":
        fast = rng.randint(4, 16)
        return {"fast": float(fast), "slow": float(rng.randint(fast + 4, 60))}
    if strategy == "volume_breakout":
        return {
            "lookback": float(rng.randint(8, 40)),
            "volume_multiple": rng.uniform(1.1, 3.0),
        }
    if strategy == "mean_reversion":
        return {
            "lookback": float(rng.randint(8, 50)),
            "z_entry": rng.uniform(0.8, 2.5),
        }
    return {
        "lookback": float(rng.randint(12, 60)),
        "pullback_ema": float(rng.randint(4, 20)),
    }


def _random_profile(rng: random.Random, generation: int, index: int) -> StrategyProfile:
    strategy = rng.choice(sorted(STRATEGY_FAMILIES))
    execution = rng.choice(sorted(EXECUTION_POLICIES))
    params = _random_params(rng, strategy)
    params["limit_atr"] = rng.uniform(0.10, 0.50)
    params["breakout_atr"] = rng.uniform(0.02, 0.25)
    return StrategyProfile(
        account_id=f"g{generation:02d}-a{index:02d}",
        strategy=strategy,
        execution_policy=execution,
        params=params,
        risk_per_trade=rng.uniform(0.005, 0.02),
        max_position_fraction=rng.uniform(0.35, 0.90),
        max_daily_loss_fraction=rng.uniform(0.02, 0.04),
        stop_atr=rng.uniform(1.0, 2.5),
        reward_to_risk=rng.uniform(1.2, 3.0),
        max_hold_bars=rng.randint(5, 40),
    )


def _mutate_profile(
    rng: random.Random,
    parent: StrategyProfile,
    generation: int,
    index: int,
) -> StrategyProfile:
    params = {
        key: max(float(value) * rng.uniform(0.80, 1.20), 0.01)
        for key, value in parent.params.items()
    }
    execution = parent.execution_policy
    if rng.random() < 0.20:
        execution = rng.choice(sorted(EXECUTION_POLICIES))
    return replace(
        parent,
        account_id=f"g{generation:02d}-m{index:02d}",
        execution_policy=execution,
        params=params,
        risk_per_trade=min(max(parent.risk_per_trade * rng.uniform(0.80, 1.20), 0.005), 0.02),
        max_position_fraction=min(
            max(parent.max_position_fraction * rng.uniform(0.85, 1.15), 0.30), 0.90
        ),
        stop_atr=min(max(parent.stop_atr * rng.uniform(0.85, 1.15), 0.8), 3.0),
        reward_to_risk=min(
            max(parent.reward_to_risk * rng.uniform(0.85, 1.15), 1.0), 3.5
        ),
        max_hold_bars=min(max(int(round(parent.max_hold_bars * rng.uniform(0.8, 1.2))), 3), 60),
    )


def _build_generation(
    rng: random.Random,
    *,
    generation: int,
    count: int,
    elites: list[StrategyProfile],
) -> list[StrategyProfile]:
    profiles: list[StrategyProfile] = []
    if elites:
        elite_limit = min(len(elites), max(2, count // 4))
        for index, elite in enumerate(elites[:elite_limit]):
            profiles.append(
                replace(elite, account_id=f"g{generation:02d}-elite{index:02d}")
            )
        while len(profiles) < count * 3 // 4:
            parent = rng.choice(elites)
            profiles.append(_mutate_profile(rng, parent, generation, len(profiles)))
    while len(profiles) < count:
        profiles.append(_random_profile(rng, generation, len(profiles)))
    return profiles


def _build_finalists(
    rng: random.Random,
    elites: list[StrategyProfile],
    target_count: int,
) -> list[StrategyProfile]:
    finalists = [
        replace(profile, account_id=f"final-{index:02d}")
        for index, profile in enumerate(elites)
    ]
    while len(finalists) < max(4, min(target_count, 12)):
        finalists.append(_random_profile(rng, 99, len(finalists)))
    return finalists


def _finite_metric(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"locked finalist metric {key} is not finite") from None
    if not math.isfinite(value):
        raise ValueError(f"locked finalist metric {key} is not finite")
    return value


def _report_summary(
    report: Any,
    profile: StrategyProfile | None = None,
    *,
    evaluation_role: str | None = None,
) -> dict[str, Any]:
    payload = {
        "account_id": report.account_id,
        "strategy": report.strategy,
        "execution_policy": report.execution_policy,
        "ending_equity": report.ending_equity,
        "total_return": report.total_return,
        "max_drawdown": report.max_drawdown,
        "sharpe": report.sharpe,
        "profit_factor": report.profit_factor,
        "win_rate": report.win_rate,
        "expectancy": report.expectancy,
        "trade_count": report.trade_count,
        "average_execution_cost": report.average_execution_cost,
        "score": report.score,
        "params": dict(profile.params) if profile is not None else None,
        "risk": {
            "risk_per_trade": profile.risk_per_trade,
            "max_position_fraction": profile.max_position_fraction,
            "max_daily_loss_fraction": profile.max_daily_loss_fraction,
            "stop_atr": profile.stop_atr,
            "reward_to_risk": profile.reward_to_risk,
            "max_hold_bars": profile.max_hold_bars,
        } if profile is not None else None,
    }
    if evaluation_role is not None:
        payload["evaluation_role"] = evaluation_role
    return _json_safe(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = ["LabJob", "LabRunSpec", "PaperLabAutomationService"]
