#!/usr/bin/env python3
"""Keep the v3 historical research loop running until it is ready.

This controller is deliberately research-only.  It runs the existing causal
v3 backtest at the two requested order sizes, persists a small state file, and
returns to a waiting state when the data and code have not changed.  It never
changes an active portfolio, stages a candidate, enables leverage, contacts a
broker, or places an order.

For a long-running local process use ``--until-ready``.  For a scheduler such
as GitHub Actions, invoke the controller once per scheduled run; a new data
fingerprint causes a fresh research iteration.  ``--force`` is available for
deliberate reruns against unchanged inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Works as ``python scripts/...`` and when imported by pytest.
    from scripts.run_momentum_volatility_research import Bar, load_bars
    from scripts.run_momentum_volatility_v3 import align_pair, research
except ModuleNotFoundError:  # pragma: no cover - direct import fallback
    from run_momentum_volatility_research import Bar, load_bars
    from run_momentum_volatility_v3 import align_pair, research


SCHEMA_VERSION = 2
DEFAULT_MIN_CONFIRMATION_ENTRIES = 5
DEFAULT_MAX_DATA_AGE_DAYS = 45.0
REQUIRED_ORDER_SIZE_KEYS = frozenset({"4000", "6000"})
DEFAULT_BTC_PATH = Path("data/historical/binance/normalized/BTCUSDT_1h.csv")
DEFAULT_ETH_PATH = Path("data/historical/binance/normalized/ETHUSDT_1h.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/momentum-v3/research-controller")
DEFAULT_MIN_ALIGNED_BARS = 3 * 365 * 24
DEFAULT_ORDER_NOTIONALS = (4_000.0, 6_000.0)
DEFAULT_MIN_FULL_ENTRIES = 8
DEFAULT_MIN_FOLD_ENTRIES = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON without leaving a half-written state file behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "researching",
            "strategy_ready": False,
            "iterations_completed": 0,
            "consecutive_ready_passes": 0,
            "history": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"research state must be a JSON object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        # Never trust readiness from an older state format. Reset to a clean
        # research state so an optional artifact restore cannot block the
        # monthly job or carry an old winner across changed gate semantics.
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "researching",
            "strategy_ready": False,
            "iterations_completed": 0,
            "consecutive_ready_passes": 0,
            "history": [],
            "migration_notice": (
                f"discarded unsupported research state schema "
                f"{payload.get('schema_version')!r}"
            ),
        }
    if not isinstance(payload.get("strategy_ready", False), bool):
        raise ValueError("research state strategy_ready must be boolean")
    if (
        not isinstance(payload.get("iterations_completed", 0), int)
        or isinstance(payload.get("iterations_completed", 0), bool)
        or payload.get("iterations_completed", 0) < 0
    ):
        raise ValueError("research state iterations_completed must be a non-negative integer")
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("research state history must be a list")
    payload.setdefault("consecutive_ready_passes", 0)
    payload.setdefault("last_ready_signature", None)
    payload.setdefault("last_ready_experiment", None)
    payload.setdefault("last_ready_candidates", [])
    consecutive = payload["consecutive_ready_passes"]
    if not isinstance(consecutive, int) or isinstance(consecutive, bool) or consecutive < 0:
        raise ValueError("research state consecutive_ready_passes must be non-negative")
    for field in ("last_ready_signature", "last_ready_experiment"):
        if payload[field] is not None and not isinstance(payload[field], Mapping):
            raise ValueError(f"research state {field} must be an object or null")
    if (
        not isinstance(payload["last_ready_candidates"], list)
        or not all(isinstance(item, str) for item in payload["last_ready_candidates"])
    ):
        raise ValueError("research state last_ready_candidates must be a string list")
    if payload.get("strategy_ready"):
        last_run = payload.get("last_run")
        readiness = last_run.get("readiness") if isinstance(last_run, Mapping) else None
        if (
            not isinstance(last_run, Mapping)
            or last_run.get("strategy_ready") is not True
            or not isinstance(readiness, Mapping)
            or readiness.get("strategy_ready") is not True
        ):
            # A cached flag without the complete evidence record is not
            # sufficient to report readiness after an artifact restore.
            payload["strategy_ready"] = False
            payload["status"] = "researching"
            payload["consecutive_ready_passes"] = 0
            payload["last_ready_signature"] = None
            payload["last_ready_experiment"] = None
            payload["last_ready_candidates"] = []
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> dict[str, str]:
    """Invalidate cached research when code or workflow inputs change."""

    files = [
        Path(__file__),
        Path(__file__).with_name("run_momentum_volatility_v3.py"),
        Path(__file__).with_name("run_momentum_volatility_research.py"),
        Path(__file__).with_name("momentum_context.py"),
        Path(__file__).with_name("fetch_binance_vision_klines.py"),
        Path(__file__).parents[1] / ".github" / "workflows" / "momentum-v3-research.yml",
    ]
    return {
        str(path): _sha256(path)
        for path in files
        if path.is_file()
    }


def _data_fingerprint(paths: Sequence[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        stat = path.stat()
        result[str(path)] = {
            "sha256": _sha256(path),
            "bytes": stat.st_size,
        }
    return result


def _gap_details(bars: Sequence[Bar]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for previous, current in zip(bars, bars[1:]):
        hours = (current.timestamp - previous.timestamp).total_seconds() / 3600.0
        if hours > 1.5:
            details.append(
                {
                    "from": previous.timestamp.isoformat(),
                    "to": current.timestamp.isoformat(),
                    "hours": hours,
                }
            )
    return details


def validate_data(
    btc_path: Path,
    eth_path: Path,
    *,
    minimum_aligned_bars: int = DEFAULT_MIN_ALIGNED_BARS,
    max_data_age_days: float | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Load both markets and fail closed on unsafe or stale research data."""

    btc = load_bars(btc_path)
    eth = load_bars(eth_path)
    pair = align_pair(btc, eth)
    if len(pair) < minimum_aligned_bars:
        raise ValueError(
            f"aligned data has {len(pair)} bars; at least {minimum_aligned_bars} are required"
        )
    if (
        btc[0].timestamp != eth[0].timestamp
        or btc[-1].timestamp != eth[-1].timestamp
        or len(pair) != len(btc)
        or len(pair) != len(eth)
    ):
        raise ValueError(
            "BTC and ETH must have identical timestamp coverage; "
            "refusing silently truncated pair data"
        )
    btc_gaps = _gap_details(btc)
    eth_gaps = _gap_details(eth)
    if [(item["from"], item["to"]) for item in btc_gaps] != [
        (item["from"], item["to"]) for item in eth_gaps
    ]:
        raise ValueError(
            "historical data contains unsynchronized gaps; refusing to compare "
            f"BTC={btc_gaps} with ETH={eth_gaps}"
        )
    largest_gap = max(
        (float(item["hours"]) for item in (*btc_gaps, *eth_gaps)),
        default=0.0,
    )
    if largest_gap > 6.0:
        raise ValueError(
            f"historical data contains a synchronized gap of {largest_gap:.2f} hours; "
            "gaps over six hours require a separately reviewed dataset"
        )
    data_age_days = None
    if max_data_age_days is not None:
        if not math.isfinite(max_data_age_days) or max_data_age_days <= 0:
            raise ValueError("max_data_age_days must be finite and positive")
        now = datetime.now(timezone.utc)
        data_age_days = (now - pair[-1].timestamp).total_seconds() / 86400.0
        if data_age_days < 0 or data_age_days > max_data_age_days:
            raise ValueError(
                f"historical data ends {data_age_days:.2f} days from the current UTC time; "
                f"maximum allowed age is {max_data_age_days:.2f} days"
            )
    return pair, {
        "btc_bars": len(btc),
        "eth_bars": len(eth),
        "aligned_bars": len(pair),
        "start": pair[0].timestamp.isoformat(),
        "end": pair[-1].timestamp.isoformat(),
        "data_age_days": data_age_days,
        "btc_gaps_over_1_5_hours": btc_gaps,
        "eth_gaps_over_1_5_hours": eth_gaps,
        "synchronized_gap_warning": bool(btc_gaps),
        "largest_gap_hours": largest_gap,
    }


def _result_number(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = float(payload.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan
    return value


def _result_integer(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = int(payload.get(key, -1))
    except (TypeError, ValueError):
        return -1
    return value


def _median_result_return(results: Sequence[Any]) -> float:
    values: list[float] = []
    for result in results:
        if not isinstance(result, Mapping):
            return math.nan
        value = _result_number(result, "return_pct")
        if not math.isfinite(value):
            return math.nan
        values.append(value)
    return statistics.median(values) if values else math.nan


def _result_reasons(
    label: str,
    result: Any,
    *,
    minimum_entries: int,
    require_positive: bool,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(result, Mapping):
        return [f"missing {label} result"]
    entries = _result_integer(result, "entries")
    if entries < minimum_entries:
        reasons.append(
            f"{label} has {entries} entries; {minimum_entries} required"
        )
    return_pct = _result_number(result, "return_pct")
    if not math.isfinite(return_pct):
        reasons.append(f"{label} return is not finite")
    elif require_positive and return_pct <= 0:
        reasons.append(f"{label} return is not positive")
    permanent_halt = result.get("permanent_halt")
    if not isinstance(permanent_halt, bool):
        reasons.append(f"{label} permanent_halt flag is invalid")
    elif permanent_halt:
        reasons.append(f"{label} hit a permanent halt")
    kill_events = _result_integer(result, "kill_events")
    if kill_events < 0:
        reasons.append(f"{label} kill_events field is invalid")
    elif kill_events > 1:
        reasons.append(f"{label} has repeated kill-switch events")
    return reasons


def _validate_report_costs(report: Mapping[str, Any]) -> list[str]:
    costs = report.get("costs")
    if not isinstance(costs, Mapping):
        return ["missing cost assumptions"]
    base = costs.get("base")
    stress = costs.get("stress")
    if not isinstance(base, Mapping) or not isinstance(stress, Mapping):
        return ["base and stress cost assumptions are required"]
    base_fees = _result_number(base, "fees_bps")
    base_slippage = _result_number(base, "slippage_bps")
    stress_fees = _result_number(stress, "fees_bps")
    stress_slippage = _result_number(stress, "slippage_bps")
    values = (base_fees, base_slippage, stress_fees, stress_slippage)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        return ["cost assumptions must be finite and non-negative"]
    if (
        stress_fees < base_fees
        or stress_slippage < base_slippage
        or (stress_fees == base_fees and stress_slippage == base_slippage)
    ):
        return [
            "stress costs must be no lower in either component and higher in at least one"
        ]
    return []


def evaluate_candidate(
    candidate_name: str,
    candidate: Mapping[str, Any],
    *,
    minimum_full_entries: int = DEFAULT_MIN_FULL_ENTRIES,
    minimum_fold_entries: int = DEFAULT_MIN_FOLD_ENTRIES,
    minimum_confirmation_entries: int = DEFAULT_MIN_CONFIRMATION_ENTRIES,
) -> dict[str, Any]:
    """Apply independent, fail-closed readiness checks to one candidate."""

    reasons: list[str] = []
    gate = candidate.get("promotion_gate")
    if not isinstance(gate, Mapping) or gate.get("pass") is not True:
        gate_reasons = gate.get("failure_reasons") if isinstance(gate, Mapping) else None
        if isinstance(gate_reasons, list) and gate_reasons:
            reasons.extend(str(item) for item in gate_reasons)
        else:
            reasons.append("v3 promotion gate did not pass a strict boolean check")

    full = candidate.get("full_sample")
    stress_full = candidate.get("full_sample_stress")
    reasons.extend(_result_reasons(
        "full sample", full, minimum_entries=minimum_full_entries, require_positive=True
    ))
    reasons.extend(_result_reasons(
        "stress full sample", stress_full, minimum_entries=minimum_full_entries, require_positive=True
    ))

    fold_checks: dict[str, list[dict[str, Any]]] = {}
    fold_medians: dict[str, float] = {}
    for label in ("walk_forward", "stress_walk_forward"):
        raw_folds = candidate.get(label)
        if (
            not isinstance(raw_folds, Sequence)
            or isinstance(raw_folds, (str, bytes))
            or len(raw_folds) != 3
        ):
            reasons.append(f"{label} must contain exactly three folds")
            fold_checks[label] = []
            fold_medians[label] = math.nan
            continue
        checks: list[dict[str, Any]] = []
        for index, raw_fold in enumerate(raw_folds, start=1):
            fold_reasons = _result_reasons(
                f"{label} fold {index}",
                raw_fold,
                minimum_entries=minimum_fold_entries,
                require_positive=False,
            )
            reasons.extend(fold_reasons)
            fold = raw_fold if isinstance(raw_fold, Mapping) else {}
            checks.append({
                "fold": index,
                "entries": _result_integer(fold, "entries"),
                "return_pct": _result_number(fold, "return_pct"),
                "pass": not fold_reasons,
                "failure_reasons": fold_reasons,
            })
        fold_checks[label] = checks
        fold_medians[label] = _median_result_return(raw_folds)

    base_median = fold_medians["walk_forward"]
    stress_median = fold_medians["stress_walk_forward"]
    if not math.isfinite(base_median) or base_median <= 0:
        reasons.append("base walk-forward median return is not positive and finite")
    if not math.isfinite(stress_median) or stress_median <= 0:
        reasons.append("stress walk-forward median return is not positive and finite")
    if isinstance(gate, Mapping):
        for field, computed in (
            ("base_walk_forward_median_return_pct", base_median),
            ("stress_walk_forward_median_return_pct", stress_median),
        ):
            reported = _result_number(gate, field)
            if not math.isfinite(reported) or not math.isfinite(computed):
                reasons.append(f"promotion gate field {field} is not finite")
            elif not math.isclose(reported, computed, rel_tol=1e-9, abs_tol=1e-9):
                reasons.append(f"promotion gate field {field} disagrees with raw folds")

    holdout = candidate.get("confirmation_holdout")
    holdout_checks: dict[str, Any] = {}
    if not isinstance(holdout, Mapping):
        reasons.append("missing confirmation holdout")
    else:
        for label in ("base", "stress"):
            holdout_checks[label] = _result_reasons(
                f"confirmation holdout {label}",
                holdout.get(label),
                minimum_entries=minimum_confirmation_entries,
                require_positive=True,
            )
            reasons.extend(holdout_checks[label])

    horizons = candidate.get("horizons")
    horizon_observations: dict[str, Any] = {}
    if not isinstance(horizons, Mapping):
        reasons.append("missing 1d/1w/1m/1y horizon evidence")
    else:
        for label in ("1d", "1w", "1m", "1y"):
            value = horizons.get(label)
            if not isinstance(value, Mapping):
                reasons.append(f"missing {label} horizon evidence")
                continue
            return_pct = _result_number(value, "return_pct")
            pnl = _result_number(value, "pnl")
            entries = _result_integer(value, "entries")
            if not math.isfinite(return_pct) or not math.isfinite(pnl) or entries < 0:
                reasons.append(f"{label} horizon evidence is invalid")
            horizon_observations[label] = {
                "return_pct": return_pct,
                "entries": entries,
                "pnl": pnl,
            }

    return {
        "candidate": candidate_name,
        "pass": not reasons,
        "failure_reasons": reasons,
        "fold_checks": fold_checks,
        "walk_forward_medians": {
            "base_return_pct": base_median,
            "stress_return_pct": stress_median,
        },
        "confirmation_holdout_checks": holdout_checks,
        "horizon_observations": horizon_observations,
        "rules": {
            "minimum_full_sample_entries": minimum_full_entries,
            "minimum_entries_per_walk_forward_fold": minimum_fold_entries,
            "minimum_confirmation_holdout_entries": minimum_confirmation_entries,
            "positive_full_sample_base_and_stress": True,
            "positive_base_and_stress_walk_forward_medians": True,
            "positive_confirmation_holdout_base_and_stress": True,
            "one_day_snapshot_is_not_sufficient_proof": True,
        },
    }


def evaluate_readiness(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    minimum_full_entries: int = DEFAULT_MIN_FULL_ENTRIES,
    minimum_fold_entries: int = DEFAULT_MIN_FOLD_ENTRIES,
    minimum_confirmation_entries: int = DEFAULT_MIN_CONFIRMATION_ENTRIES,
) -> dict[str, Any]:
    """Require one candidate to pass the complete gate at exactly both sizes."""

    normalized_reports = {str(size): report for size, report in reports.items()}
    global_reasons: list[str] = []
    if set(normalized_reports) != set(REQUIRED_ORDER_SIZE_KEYS):
        global_reasons.append("readiness requires exactly $4,000 and $6,000 reports")

    checks_by_size: dict[str, dict[str, Any]] = {}
    candidate_sets: list[set[str]] = []
    for size in sorted(REQUIRED_ORDER_SIZE_KEYS):
        report = normalized_reports.get(size)
        if not isinstance(report, Mapping):
            global_reasons.append(f"missing report for order size {size}")
            report = {}
        global_reasons.extend(
            f"${size}: {reason}" for reason in _validate_report_costs(report)
        )
        raw_candidates = report.get("candidates", {})
        if not isinstance(raw_candidates, Mapping):
            global_reasons.append(f"${size}: missing candidates mapping")
            raw_candidates = {}
        checks = {
            str(name): evaluate_candidate(
                str(name),
                candidate,
                minimum_full_entries=minimum_full_entries,
                minimum_fold_entries=minimum_fold_entries,
                minimum_confirmation_entries=minimum_confirmation_entries,
            )
            for name, candidate in raw_candidates.items()
            if isinstance(candidate, Mapping)
        }
        checks_by_size[size] = checks
        candidate_sets.append(set(checks))

    common_candidates = set.intersection(*candidate_sets) if len(candidate_sets) == 2 else set()
    ready_candidates = sorted(
        name
        for name in common_candidates
        if all(bool(checks_by_size[size][name]["pass"]) for size in REQUIRED_ORDER_SIZE_KEYS)
    )
    if global_reasons:
        ready_candidates = []
    return {
        "pass": bool(ready_candidates),
        "strategy_ready": bool(ready_candidates),
        "ready_candidates": ready_candidates,
        "required_sizes": sorted(REQUIRED_ORDER_SIZE_KEYS),
        "common_candidates": sorted(common_candidates),
        "checks_by_size": checks_by_size,
        "failure_reasons": global_reasons,
        "rules": {
            "same_candidate_must_pass_all_required_sizes": True,
            "required_order_notionals": [4_000.0, 6_000.0],
            "required_confirmation_holdout": True,
            "required_distinct_data_vintages": 3,
            "leverage_allowed": False,
            "paper_promotion_is_not_automatic": True,
        },
    }


def _signature(
    data: Mapping[str, Any],
    source: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "data": dict(data),
        "source": dict(source),
        "config": dict(config),
    }


def _save_status(state_path: Path, state: Mapping[str, Any]) -> None:
    _atomic_write_json(state_path, state)
    _atomic_write_json(state_path.with_name("latest.json"), state)


def _run_iteration(
    *,
    iteration: int,
    btc_path: Path,
    eth_path: Path,
    output_dir: Path,
    minimum_aligned_bars: int,
    order_notionals: Sequence[float],
    research_initial_balance: float,
    fees_bps: float,
    slippage_bps: float,
    stress_fees_bps: float,
    stress_slippage_bps: float,
    horizon_initial_balance: float,
    context_path: Path | None,
    minimum_full_entries: int,
    minimum_fold_entries: int,
    minimum_confirmation_entries: int,
    max_data_age_days: float | None,
    research_config: Mapping[str, Any],
    data_fingerprint: Mapping[str, Any],
    source_fingerprint: Mapping[str, str],
) -> dict[str, Any]:
    started_at = _utc_now()
    pair, quality = validate_data(
        btc_path,
        eth_path,
        minimum_aligned_bars=minimum_aligned_bars,
        max_data_age_days=max_data_age_days,
    )
    del pair  # validation above deliberately reloads inside the v3 runner.

    iteration_dir = output_dir / f"iteration-{iteration:04d}"
    reports: dict[str, Mapping[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for order_notional in order_notionals:
        size_key = str(int(order_notional))
        report = research(
            btc_path,
            eth_path,
            initial_balance=research_initial_balance,
            order_notional=order_notional,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            stress_fees_bps=stress_fees_bps,
            stress_slippage_bps=stress_slippage_bps,
            horizon_initial_balance=horizon_initial_balance,
            horizon_order_notional=order_notional,
            context_path=context_path,
        )
        report["controller"] = {
            "schema_version": SCHEMA_VERSION,
            "iteration": iteration,
            "research_only": True,
            "active_profile_changed": False,
            "data_quality": quality,
            "data_fingerprint": dict(data_fingerprint),
            "source_fingerprint": dict(source_fingerprint),
            "research_config": dict(research_config),
        }
        reports[size_key] = report
        report_path = iteration_dir / f"report-{size_key}.json"
        _atomic_write_json(report_path, report)
        report_paths[size_key] = str(report_path)

    readiness = evaluate_readiness(
        reports,
        minimum_full_entries=minimum_full_entries,
        minimum_fold_entries=minimum_fold_entries,
        minimum_confirmation_entries=minimum_confirmation_entries,
    )
    finished_at = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if readiness["strategy_ready"] else "researching",
        "strategy_ready": bool(readiness["strategy_ready"]),
        "iteration": iteration,
        "started_at": started_at,
        "finished_at": finished_at,
        "data_quality": quality,
        "data_fingerprint": dict(data_fingerprint),
        "source_fingerprint": dict(source_fingerprint),
        "readiness": readiness,
        "report_paths": report_paths,
        "active_profile_changed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
    }


def _append_history(state: dict[str, Any], entry: Mapping[str, Any]) -> None:
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        state["history"] = history
    history.append(dict(entry))
    del history[:-1000]


def run_controller(
    *,
    btc_path: Path = DEFAULT_BTC_PATH,
    eth_path: Path = DEFAULT_ETH_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    state_path: Path | None = None,
    minimum_aligned_bars: int = DEFAULT_MIN_ALIGNED_BARS,
    order_notionals: Sequence[float] = DEFAULT_ORDER_NOTIONALS,
    research_initial_balance: float = 75_000.0,
    fees_bps: float = 10.0,
    slippage_bps: float = 5.0,
    stress_fees_bps: float = 20.0,
    stress_slippage_bps: float = 10.0,
    horizon_initial_balance: float = 5_000.0,
    context_path: Path | None = None,
    minimum_full_entries: int = DEFAULT_MIN_FULL_ENTRIES,
    minimum_fold_entries: int = DEFAULT_MIN_FOLD_ENTRIES,
    minimum_confirmation_entries: int = DEFAULT_MIN_CONFIRMATION_ENTRIES,
    max_data_age_days: float | None = DEFAULT_MAX_DATA_AGE_DAYS,
    required_consecutive_passes: int = 3,
    until_ready: bool = False,
    interval_hours: float = 24.0,
    max_iterations: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    if len(order_notionals) != 2 or set(order_notionals) != {4_000.0, 6_000.0}:
        raise ValueError("the controller requires exactly $4,000 and $6,000 order notionals")
    if any(not math.isfinite(value) or value <= 0 for value in order_notionals):
        raise ValueError("order notionals must be finite and positive")
    if (
        minimum_aligned_bars < 1
        or minimum_full_entries < 1
        or minimum_fold_entries < 1
        or minimum_confirmation_entries < 1
    ):
        raise ValueError("research minimums must be positive")
    if required_consecutive_passes < 1:
        raise ValueError("required_consecutive_passes must be positive")
    cost_values = (fees_bps, slippage_bps, stress_fees_bps, stress_slippage_bps)
    if not all(math.isfinite(value) and value >= 0 for value in cost_values):
        raise ValueError("cost assumptions must be finite and non-negative")
    if (
        stress_fees_bps < fees_bps
        or stress_slippage_bps < slippage_bps
        or (stress_fees_bps == fees_bps and stress_slippage_bps == slippage_bps)
    ):
        raise ValueError(
            "stress costs must be no lower in either component and higher in at least one"
        )
    if max_data_age_days is not None and (
        not math.isfinite(max_data_age_days) or max_data_age_days <= 0
    ):
        raise ValueError("max_data_age_days must be finite and positive")
    if until_ready and max_iterations == 0:
        raise ValueError("--until-ready requires a finite --max-iterations safety bound")
    if until_ready and (not math.isfinite(interval_hours) or interval_hours < 0):
        raise ValueError("interval_hours must be finite and non-negative")
    if max_iterations < 0:
        raise ValueError("max_iterations cannot be negative")

    btc_path = btc_path.resolve()
    eth_path = eth_path.resolve()
    context_path = context_path.resolve() if context_path else None
    output_dir = output_dir.resolve()
    state_path = (state_path or output_dir / "state.json").resolve()
    if not btc_path.is_file() or not eth_path.is_file():
        raise FileNotFoundError(f"both BTC and ETH CSVs are required: {btc_path}, {eth_path}")
    if context_path is not None and not context_path.is_file():
        raise FileNotFoundError(f"context CSV does not exist: {context_path}")

    research_config = {
        "schema_version": SCHEMA_VERSION,
        "minimum_aligned_bars": minimum_aligned_bars,
        "order_notionals": [float(value) for value in order_notionals],
        "research_initial_balance": research_initial_balance,
        "fees_bps": fees_bps,
        "slippage_bps": slippage_bps,
        "stress_fees_bps": stress_fees_bps,
        "stress_slippage_bps": stress_slippage_bps,
        "horizon_initial_balance": horizon_initial_balance,
        "context_path": str(context_path) if context_path else None,
        "minimum_full_entries": minimum_full_entries,
        "minimum_fold_entries": minimum_fold_entries,
        "minimum_confirmation_entries": minimum_confirmation_entries,
        "max_data_age_days": max_data_age_days,
        "required_consecutive_passes": required_consecutive_passes,
    }

    state = _read_state(state_path)
    source_fingerprint = _source_fingerprint()
    iterations_this_call = 0
    while True:
        data_paths = (btc_path, eth_path) + ((context_path,) if context_path else ())
        data_fingerprint = _data_fingerprint(data_paths)
        signature = _signature(data_fingerprint, source_fingerprint, research_config)
        if not force and state.get("last_signature") == signature:
            state.update({
                "schema_version": SCHEMA_VERSION,
                "status": "ready" if state.get("strategy_ready") else "waiting_for_new_data",
                "last_checked_at": _utc_now(),
                "active_profile_changed": False,
            })
            _save_status(state_path, state)
            if not until_ready:
                return state
            iterations_this_call += 1
            if max_iterations and iterations_this_call >= max_iterations:
                return state
            time.sleep(interval_hours * 3600.0)
            continue

        iteration = int(state.get("iterations_completed", 0)) + 1
        result = _run_iteration(
            iteration=iteration,
            btc_path=btc_path,
            eth_path=eth_path,
            output_dir=output_dir,
            minimum_aligned_bars=minimum_aligned_bars,
            order_notionals=order_notionals,
            research_initial_balance=research_initial_balance,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            stress_fees_bps=stress_fees_bps,
            stress_slippage_bps=stress_slippage_bps,
            horizon_initial_balance=horizon_initial_balance,
            context_path=context_path,
            minimum_full_entries=minimum_full_entries,
            minimum_fold_entries=minimum_fold_entries,
            minimum_confirmation_entries=minimum_confirmation_entries,
            max_data_age_days=max_data_age_days,
            research_config=research_config,
            data_fingerprint=data_fingerprint,
            source_fingerprint=source_fingerprint,
        )
        candidate_ready = bool(result["strategy_ready"])
        ready_candidates = list(result["readiness"]["ready_candidates"])
        experiment_signature = {"source": dict(source_fingerprint), "config": dict(research_config)}
        previous_ready_signature = state.get("last_ready_signature")
        same_data = previous_ready_signature == signature
        same_experiment = state.get("last_ready_experiment") == experiment_signature
        same_candidates = state.get("last_ready_candidates") == ready_candidates
        if candidate_ready:
            previous_count = int(state.get("consecutive_ready_passes", 0))
            consecutive_ready_passes = (
                previous_count + 1
                if same_experiment and same_candidates and not same_data
                else 1
            )
            state["last_ready_signature"] = signature
            state["last_ready_experiment"] = experiment_signature
            state["last_ready_candidates"] = ready_candidates
        else:
            consecutive_ready_passes = 0
            state["last_ready_signature"] = None
            state["last_ready_experiment"] = None
            state["last_ready_candidates"] = []
        strategy_ready = candidate_ready and consecutive_ready_passes >= required_consecutive_passes
        result["candidate_ready"] = candidate_ready
        result["strategy_ready"] = strategy_ready
        result["consecutive_ready_passes"] = consecutive_ready_passes
        result["required_consecutive_passes"] = required_consecutive_passes

        state.update({
            "schema_version": SCHEMA_VERSION,
            "status": "ready" if strategy_ready else "candidate_for_review" if candidate_ready else "researching",
            "strategy_ready": strategy_ready,
            "iterations_completed": iteration,
            "last_checked_at": _utc_now(),
            "last_run": result,
            "last_signature": signature,
            "active_profile_changed": False,
            "paper_orders_placed": False,
            "leverage_enabled": False,
            "consecutive_ready_passes": consecutive_ready_passes,
        })
        _append_history(state, {
            "iteration": iteration,
            "finished_at": result["finished_at"],
            "data_end": result["data_quality"]["end"],
            "data_signature": signature,
            "candidate_ready": candidate_ready,
            "strategy_ready": strategy_ready,
            "consecutive_ready_passes": consecutive_ready_passes,
            "ready_candidates": result["readiness"]["ready_candidates"],
            "report_paths": result["report_paths"],
        })
        _save_status(state_path, state)
        iterations_this_call += 1
        force = False
        if strategy_ready or not until_ready:
            return state
        if max_iterations and iterations_this_call >= max_iterations:
            return state
        time.sleep(interval_hours * 3600.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, default=DEFAULT_BTC_PATH)
    parser.add_argument("--eth-path", type=Path, default=DEFAULT_ETH_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state", type=Path, help="state JSON; defaults to <output-dir>/state.json")
    parser.add_argument("--min-aligned-bars", type=int, default=DEFAULT_MIN_ALIGNED_BARS)
    parser.add_argument("--research-initial-balance", type=float, default=75_000.0)
    parser.add_argument("--fees-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stress-fees-bps", type=float, default=20.0)
    parser.add_argument("--stress-slippage-bps", type=float, default=10.0)
    parser.add_argument("--horizon-initial-balance", type=float, default=5_000.0)
    parser.add_argument("--context-csv", type=Path, help="optional timestamp,sentiment,impact CSV")
    parser.add_argument("--min-full-entries", type=int, default=DEFAULT_MIN_FULL_ENTRIES)
    parser.add_argument("--min-fold-entries", type=int, default=DEFAULT_MIN_FOLD_ENTRIES)
    parser.add_argument("--min-confirmation-entries", type=int, default=DEFAULT_MIN_CONFIRMATION_ENTRIES)
    parser.add_argument("--max-data-age-days", type=float, default=DEFAULT_MAX_DATA_AGE_DAYS)
    parser.add_argument(
        "--required-consecutive-passes",
        type=int,
        default=3,
        help="distinct data vintages required before strategy_ready becomes true",
    )
    parser.add_argument(
        "--until-ready",
        action="store_true",
        help="continue polling until the same candidate passes the required data vintages",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=24.0,
        help="sleep between checks while --until-ready is active",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="optional safety bound for one invocation; 0 means no bound",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even when the data and research code are unchanged",
    )
    args = parser.parse_args()
    state = run_controller(
        btc_path=args.btc_path,
        eth_path=args.eth_path,
        output_dir=args.output_dir,
        state_path=args.state,
        minimum_aligned_bars=args.min_aligned_bars,
        research_initial_balance=args.research_initial_balance,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        stress_fees_bps=args.stress_fees_bps,
        stress_slippage_bps=args.stress_slippage_bps,
        horizon_initial_balance=args.horizon_initial_balance,
        context_path=args.context_csv,
        minimum_full_entries=args.min_full_entries,
        minimum_fold_entries=args.min_fold_entries,
        minimum_confirmation_entries=args.min_confirmation_entries,
        max_data_age_days=args.max_data_age_days,
        required_consecutive_passes=args.required_consecutive_passes,
        until_ready=args.until_ready,
        interval_hours=args.interval_hours,
        max_iterations=args.max_iterations,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": state.get("status"),
                "strategy_ready": state.get("strategy_ready", False),
                "iterations_completed": state.get("iterations_completed", 0),
                "ready_candidates": state.get("last_run", {})
                .get("readiness", {})
                .get("ready_candidates", []),
                "latest_state": str((args.state or args.output_dir / "state.json").resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
