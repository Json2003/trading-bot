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


SCHEMA_VERSION = 1
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "researching",
            "strategy_ready": False,
            "iterations_completed": 0,
            "history": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"research state must be a JSON object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported research state schema: {payload.get('schema_version')!r}"
        )
    payload.setdefault("history", [])
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> dict[str, str]:
    """Invalidate a cached iteration when the research implementation changes."""

    files = (
        Path(__file__),
        Path(__file__).with_name("run_momentum_volatility_v3.py"),
        Path(__file__).with_name("run_momentum_volatility_research.py"),
    )
    return {str(path): _sha256(path) for path in files}


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
) -> tuple[list[Any], dict[str, Any]]:
    """Load both markets and fail closed on short or unsafe research data.

    A synchronized exchange-wide outage can be retained without inventing
    candles.  Asymmetric or very large gaps are rejected because they can
    change the leader and volatility calculations differently by asset.
    """

    btc = load_bars(btc_path)
    eth = load_bars(eth_path)
    pair = align_pair(btc, eth)
    if len(pair) < minimum_aligned_bars:
        raise ValueError(
            f"aligned data has {len(pair)} bars; at least {minimum_aligned_bars} are required"
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
    return pair, {
        "btc_bars": len(btc),
        "eth_bars": len(eth),
        "aligned_bars": len(pair),
        "start": pair[0].timestamp.isoformat(),
        "end": pair[-1].timestamp.isoformat(),
        "btc_gaps_over_1_5_hours": btc_gaps,
        "eth_gaps_over_1_5_hours": eth_gaps,
        "synchronized_gap_warning": bool(btc_gaps),
        "largest_gap_hours": largest_gap,
    }


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def evaluate_candidate(
    candidate_name: str,
    candidate: Mapping[str, Any],
    *,
    minimum_full_entries: int = DEFAULT_MIN_FULL_ENTRIES,
    minimum_fold_entries: int = DEFAULT_MIN_FOLD_ENTRIES,
) -> dict[str, Any]:
    """Apply the stricter controller gate to one v3 candidate report.

    The v3 runner's gate is necessary but intentionally broad.  The
    controller additionally requires enough entries in every chronological
    walk-forward fold and a positive full-sample result.  A one-day snapshot
    is reported as evidence, not treated as proof of robustness.
    """

    reasons: list[str] = []
    gate = candidate.get("promotion_gate")
    if not isinstance(gate, Mapping):
        reasons.append("missing v3 promotion gate")
    elif not bool(gate.get("pass")):
        gate_reasons = gate.get("failure_reasons")
        if isinstance(gate_reasons, list) and gate_reasons:
            reasons.extend(str(item) for item in gate_reasons)
        else:
            reasons.append("v3 promotion gate did not pass")

    full = candidate.get("full_sample")
    if not isinstance(full, Mapping):
        reasons.append("missing full-sample result")
        full = {}
    if _integer(full, "entries") < minimum_full_entries:
        reasons.append(f"fewer than {minimum_full_entries} full-sample entries")
    if _number(full, "return_pct") <= 0:
        reasons.append("full-sample return is not positive")
    if bool(full.get("permanent_halt")):
        reasons.append("full sample hit a permanent halt")
    if _integer(full, "kill_events") > 1:
        reasons.append("full sample has repeated kill-switch events")

    fold_checks: dict[str, list[dict[str, Any]]] = {}
    for label in ("walk_forward", "stress_walk_forward"):
        raw_folds = candidate.get(label)
        if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, (str, bytes)):
            reasons.append(f"missing {label} results")
            fold_checks[label] = []
            continue
        checks: list[dict[str, Any]] = []
        for index, raw_fold in enumerate(raw_folds, start=1):
            fold = raw_fold if isinstance(raw_fold, Mapping) else {}
            entries = _integer(fold, "entries")
            fold_reasons: list[str] = []
            if entries < minimum_fold_entries:
                fold_reasons.append(
                    f"fold {index} has {entries} entries; {minimum_fold_entries} required"
                )
            if bool(fold.get("permanent_halt")):
                fold_reasons.append(f"fold {index} hit a permanent halt")
            if _integer(fold, "kill_events") > 1:
                fold_reasons.append(f"fold {index} has repeated kill-switch events")
            reasons.extend(f"{label}: {item}" for item in fold_reasons)
            checks.append(
                {
                    "fold": index,
                    "entries": entries,
                    "return_pct": _number(fold, "return_pct"),
                    "pass": not fold_reasons,
                    "failure_reasons": fold_reasons,
                }
            )
        fold_checks[label] = checks

    horizons = candidate.get("horizons")
    horizon_observations: dict[str, Any] = {}
    if isinstance(horizons, Mapping):
        for label in ("1d", "1w", "1m", "1y"):
            value = horizons.get(label)
            if isinstance(value, Mapping):
                horizon_observations[label] = {
                    "return_pct": _number(value, "return_pct"),
                    "entries": _integer(value, "entries"),
                    "pnl": _number(value, "pnl"),
                }

    return {
        "candidate": candidate_name,
        "pass": not reasons,
        "failure_reasons": reasons,
        "fold_checks": fold_checks,
        "horizon_observations": horizon_observations,
        "rules": {
            "minimum_full_sample_entries": minimum_full_entries,
            "minimum_entries_per_walk_forward_fold": minimum_fold_entries,
            "positive_full_sample_return": True,
            "positive_base_and_stress_walk_forward_medians": True,
            "one_day_snapshot_is_not_sufficient_proof": True,
        },
    }


def evaluate_readiness(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    minimum_full_entries: int = DEFAULT_MIN_FULL_ENTRIES,
    minimum_fold_entries: int = DEFAULT_MIN_FOLD_ENTRIES,
) -> dict[str, Any]:
    """Require the same candidate to pass at both $4k and $6k sizing."""

    checks_by_size: dict[str, dict[str, Any]] = {}
    candidate_sets: list[set[str]] = []
    for size, report in reports.items():
        raw_candidates = report.get("candidates", {})
        if not isinstance(raw_candidates, Mapping):
            raw_candidates = {}
        checks = {
            str(name): evaluate_candidate(
                str(name),
                candidate,
                minimum_full_entries=minimum_full_entries,
                minimum_fold_entries=minimum_fold_entries,
            )
            for name, candidate in raw_candidates.items()
            if isinstance(candidate, Mapping)
        }
        checks_by_size[str(size)] = checks
        candidate_sets.append(set(checks))

    common_candidates = sorted(set.intersection(*candidate_sets)) if candidate_sets else []
    ready_candidates = [
        name
        for name in common_candidates
        if all(bool(checks_by_size[size][name]["pass"]) for size in checks_by_size)
    ]
    return {
        "pass": bool(ready_candidates),
        "strategy_ready": bool(ready_candidates),
        "ready_candidates": ready_candidates,
        "required_sizes": sorted(checks_by_size),
        "common_candidates": common_candidates,
        "checks_by_size": checks_by_size,
        "rules": {
            "same_candidate_must_pass_all_required_sizes": True,
            "required_order_notionals": [4_000.0, 6_000.0],
            "leverage_allowed": False,
            "paper_promotion_is_not_automatic": True,
        },
    }


def _signature(data: Mapping[str, Any], source: Mapping[str, str]) -> dict[str, Any]:
    return {"data": dict(data), "source": dict(source)}


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
    minimum_full_entries: int,
    minimum_fold_entries: int,
    data_fingerprint: Mapping[str, Any],
    source_fingerprint: Mapping[str, str],
) -> dict[str, Any]:
    started_at = _utc_now()
    pair, quality = validate_data(
        btc_path,
        eth_path,
        minimum_aligned_bars=minimum_aligned_bars,
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
        )
        report["controller"] = {
            "schema_version": SCHEMA_VERSION,
            "iteration": iteration,
            "research_only": True,
            "active_profile_changed": False,
            "data_quality": quality,
            "data_fingerprint": dict(data_fingerprint),
            "source_fingerprint": dict(source_fingerprint),
        }
        reports[size_key] = report
        report_path = iteration_dir / f"report-{size_key}.json"
        _atomic_write_json(report_path, report)
        report_paths[size_key] = str(report_path)

    readiness = evaluate_readiness(
        reports,
        minimum_full_entries=minimum_full_entries,
        minimum_fold_entries=minimum_fold_entries,
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
    del history[:-100]


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
    minimum_full_entries: int = DEFAULT_MIN_FULL_ENTRIES,
    minimum_fold_entries: int = DEFAULT_MIN_FOLD_ENTRIES,
    until_ready: bool = False,
    interval_hours: float = 24.0,
    max_iterations: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    if len(order_notionals) != 2 or set(order_notionals) != set(DEFAULT_ORDER_NOTIONALS):
        raise ValueError("the controller requires exactly $4,000 and $6,000 order notionals")
    if any(value <= 0 for value in order_notionals):
        raise ValueError("order notionals must be positive")
    if minimum_aligned_bars < 1 or minimum_full_entries < 1 or minimum_fold_entries < 1:
        raise ValueError("research minimums must be positive")
    if until_ready and interval_hours < 0:
        raise ValueError("interval_hours cannot be negative")
    if until_ready and interval_hours == 0 and max_iterations == 0:
        raise ValueError(
            "an unlimited --until-ready loop needs a positive interval_hours"
        )
    if max_iterations < 0:
        raise ValueError("max_iterations cannot be negative")

    btc_path = btc_path.resolve()
    eth_path = eth_path.resolve()
    output_dir = output_dir.resolve()
    state_path = (state_path or output_dir / "state.json").resolve()
    if not btc_path.is_file() or not eth_path.is_file():
        raise FileNotFoundError(f"both BTC and ETH CSVs are required: {btc_path}, {eth_path}")

    state = _read_state(state_path)
    source_fingerprint = _source_fingerprint()
    iterations_this_call = 0
    while True:
        data_fingerprint = _data_fingerprint((btc_path, eth_path))
        signature = _signature(data_fingerprint, source_fingerprint)
        last_signature = state.get("last_signature")
        if not force and last_signature == signature:
            state.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ready" if state.get("strategy_ready") else "waiting_for_new_data",
                    "last_checked_at": _utc_now(),
                    "active_profile_changed": False,
                }
            )
            _save_status(state_path, state)
            if not until_ready:
                return state
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
            minimum_full_entries=minimum_full_entries,
            minimum_fold_entries=minimum_fold_entries,
            data_fingerprint=data_fingerprint,
            source_fingerprint=source_fingerprint,
        )
        state.update(
            {
                "schema_version": SCHEMA_VERSION,
                "status": result["status"],
                "strategy_ready": result["strategy_ready"],
                "iterations_completed": iteration,
                "last_checked_at": _utc_now(),
                "last_run": result,
                "last_signature": signature,
                "active_profile_changed": False,
                "paper_orders_placed": False,
                "leverage_enabled": False,
            }
        )
        _append_history(
            state,
            {
                "iteration": iteration,
                "finished_at": result["finished_at"],
                "data_end": result["data_quality"]["end"],
                "strategy_ready": result["strategy_ready"],
                "ready_candidates": result["readiness"]["ready_candidates"],
                "report_paths": result["report_paths"],
            },
        )
        _save_status(state_path, state)
        iterations_this_call += 1
        force = False

        if result["strategy_ready"] or not until_ready:
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
    parser.add_argument("--min-full-entries", type=int, default=DEFAULT_MIN_FULL_ENTRIES)
    parser.add_argument("--min-fold-entries", type=int, default=DEFAULT_MIN_FOLD_ENTRIES)
    parser.add_argument(
        "--until-ready",
        action="store_true",
        help="continue polling until a candidate passes the controller gate",
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
        minimum_full_entries=args.min_full_entries,
        minimum_fold_entries=args.min_fold_entries,
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
