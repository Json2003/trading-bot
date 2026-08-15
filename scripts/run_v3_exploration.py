#!/usr/bin/env python3
"""Bounded, research-only development screen for v3 candidates.

The screen deliberately excludes the latest one-year confirmation period from
every calculation. It can shortlist pre-registered candidates for a future,
separate confirmation run, but it never marks a strategy ready, changes an
active profile, enables leverage, or places an order.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.momentum_context import load_context_events
    from scripts.run_momentum_volatility_v3 import (
        CONFIRMATION_HOLDOUT_BARS,
        V3Config,
        build_pair_features,
        run_pair,
    )
    from scripts.run_v3_research_controller import (
        DEFAULT_MAX_DATA_AGE_DAYS,
        DEFAULT_MIN_ALIGNED_BARS,
        validate_data,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from momentum_context import load_context_events
    from run_momentum_volatility_v3 import (
        CONFIRMATION_HOLDOUT_BARS,
        V3Config,
        build_pair_features,
        run_pair,
    )
    from run_v3_research_controller import (
        DEFAULT_MAX_DATA_AGE_DAYS,
        DEFAULT_MIN_ALIGNED_BARS,
        validate_data,
    )


SUITE_VERSION = "v3-bounded-development-screen-1"
ORDER_NOTIONALS = (4_000.0, 6_000.0)
BASE_COSTS = {"fees_bps": 10.0, "slippage_bps": 5.0}
STRESS_COSTS = {"fees_bps": 20.0, "slippage_bps": 10.0}
MIN_FULL_ENTRIES = 8
MIN_FOLD_ENTRIES = 5
MAX_CANDIDATES = 12


def candidate_definitions() -> dict[str, V3Config]:
    """Return a fixed, intentionally small candidate suite.

    The suite is pre-declared in source so a run cannot adapt its search space
    after observing the result. It contains the three current candidates and
    nine distinct regime, entry, and exit hypotheses.
    """

    base = V3Config()
    candidates = {
        "balanced": replace(
            base,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            reduce_size_rank=0.75,
            extreme_vol_rank=0.90,
            trailing_stop_atr=2.5,
            time_stop_bars=72,
        ),
        "selective": replace(
            base,
            expansion_ratio=1.10,
            min_vol_rank=0.35,
            reduce_size_rank=0.75,
            extreme_vol_rank=0.90,
            trailing_stop_atr=2.5,
            time_stop_bars=72,
        ),
        "conservative": replace(
            base,
            expansion_ratio=1.10,
            min_vol_rank=0.35,
            reduce_size_rank=0.70,
            extreme_vol_rank=0.85,
            trailing_stop_atr=3.0,
            time_stop_bars=96,
            profit_lock_activation_atr=1.25,
        ),
        "early_expansion": replace(
            base,
            expansion_ratio=1.02,
            min_vol_rank=0.20,
            trailing_stop_atr=2.0,
            time_stop_bars=48,
        ),
        "fast_trend": replace(
            base,
            fast_window=5,
            slow_window=15,
            breakout_lookback=16,
            expansion_ratio=1.03,
            min_vol_rank=0.20,
            trailing_stop_atr=2.0,
            time_stop_bars=48,
        ),
        "slow_trend": replace(
            base,
            fast_window=13,
            slow_window=34,
            breakout_lookback=24,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            trailing_stop_atr=3.0,
            time_stop_bars=96,
        ),
        "high_quality_entry": replace(
            base,
            expansion_ratio=1.08,
            min_vol_rank=0.30,
            entry_min_body_atr=0.65,
            entry_volume_multiplier=1.25,
            min_expected_edge_bps=10.0,
            trailing_stop_atr=2.5,
        ),
        "strict_leader": replace(
            base,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            leader_min_score=0.10,
            leader_margin=0.10,
            trailing_stop_atr=2.5,
        ),
        "liquidity_strict": replace(
            base,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            min_context_volume_ratio=0.90,
            entry_volume_multiplier=1.25,
            trailing_stop_atr=2.5,
        ),
        "wide_trail": replace(
            base,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            hard_stop_atr=2.5,
            trailing_stop_atr=3.5,
            time_stop_bars=96,
        ),
        "tight_risk": replace(
            base,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            hard_stop_atr=1.5,
            trailing_stop_atr=2.0,
            time_stop_bars=48,
        ),
        "edge_strict": replace(
            base,
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            expected_move_atr_multiple=1.20,
            min_expected_edge_bps=12.0,
            trailing_stop_atr=2.5,
        ),
    }
    if len(candidates) != MAX_CANDIDATES:
        raise RuntimeError("bounded exploration suite was modified unexpectedly")
    return candidates


def development_folds(length: int) -> tuple[int, list[tuple[int, int]]]:
    """Create three non-overlapping folds before a protected final year."""

    confirmation_start = length - CONFIRMATION_HOLDOUT_BARS
    if confirmation_start <= 0:
        raise ValueError("data must contain a protected one-year confirmation period")
    screen_start = confirmation_start // 2
    width = (confirmation_start - screen_start) // 3
    if width < 24 * 30:
        raise ValueError("not enough pre-confirmation data for three development folds")
    folds = [
        (screen_start, screen_start + width),
        (screen_start + width, screen_start + 2 * width),
        (screen_start + 2 * width, confirmation_start),
    ]
    if any(start >= end for start, end in folds):
        raise ValueError("invalid development fold boundaries")
    return confirmation_start, folds


def _number(payload: Mapping[str, object], key: str) -> float:
    try:
        value = float(payload.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    try:
        value = int(payload.get(key, -1))
    except (TypeError, ValueError):
        return -1
    return value


def _result_reasons(
    label: str,
    result: Mapping[str, object],
    *,
    minimum_entries: int,
    require_positive: bool,
) -> list[str]:
    reasons: list[str] = []
    entries = _integer(result, "entries")
    if entries < minimum_entries:
        reasons.append(f"{label} has {entries} entries; {minimum_entries} required")
    return_pct = _number(result, "return_pct")
    if not math.isfinite(return_pct):
        reasons.append(f"{label} return is not finite")
    elif require_positive and return_pct <= 0:
        reasons.append(f"{label} return is not positive")
    if result.get("permanent_halt") is not False:
        reasons.append(f"{label} has an invalid or permanent halt state")
    kill_events = _integer(result, "kill_events")
    if kill_events < 0:
        reasons.append(f"{label} kill-switch count is invalid")
    elif kill_events > 1:
        reasons.append(f"{label} has repeated kill-switch events")
    return reasons


def _median_return(results: Sequence[Mapping[str, object]]) -> float:
    values = [_number(result, "return_pct") for result in results]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return statistics.median(values)


def screen_size(
    pair: Sequence[Any],
    config: V3Config,
    feature_map: Mapping[str, Mapping[str, list[float]]],
    *,
    order_notional: float,
    development_end: int,
    folds: Sequence[tuple[int, int]],
) -> dict[str, object]:
    """Screen one candidate at one notional without touching final confirmation."""

    base = run_pair(
        pair,
        initial_balance=75_000.0,
        order_notional=order_notional,
        fees_bps=BASE_COSTS["fees_bps"],
        slippage_bps=BASE_COSTS["slippage_bps"],
        config=config,
        end_index=development_end,
        feature_map=feature_map,
    )
    stress = run_pair(
        pair,
        initial_balance=75_000.0,
        order_notional=order_notional,
        fees_bps=STRESS_COSTS["fees_bps"],
        slippage_bps=STRESS_COSTS["slippage_bps"],
        config=config,
        end_index=development_end,
        feature_map=feature_map,
    )
    base_folds = [
        run_pair(
            pair,
            initial_balance=75_000.0,
            order_notional=order_notional,
            fees_bps=BASE_COSTS["fees_bps"],
            slippage_bps=BASE_COSTS["slippage_bps"],
            config=config,
            start_index=start,
            end_index=end,
            feature_map=feature_map,
        )
        for start, end in folds
    ]
    stress_folds = [
        run_pair(
            pair,
            initial_balance=75_000.0,
            order_notional=order_notional,
            fees_bps=STRESS_COSTS["fees_bps"],
            slippage_bps=STRESS_COSTS["slippage_bps"],
            config=config,
            start_index=start,
            end_index=end,
            feature_map=feature_map,
        )
        for start, end in folds
    ]

    reasons = _result_reasons(
        "development base", base, minimum_entries=MIN_FULL_ENTRIES, require_positive=True
    )
    reasons.extend(
        _result_reasons(
            "development stress",
            stress,
            minimum_entries=MIN_FULL_ENTRIES,
            require_positive=True,
        )
    )
    for label, results in (("base fold", base_folds), ("stress fold", stress_folds)):
        if len(results) != 3:
            reasons.append(f"{label} count is not three")
            continue
        for index, result in enumerate(results, start=1):
            reasons.extend(
                _result_reasons(
                    f"{label} {index}",
                    result,
                    minimum_entries=MIN_FOLD_ENTRIES,
                    require_positive=False,
                )
            )

    base_median = _median_return(base_folds)
    stress_median = _median_return(stress_folds)
    if not math.isfinite(base_median) or base_median <= 0:
        reasons.append("base development-fold median is not positive and finite")
    if not math.isfinite(stress_median) or stress_median <= 0:
        reasons.append("stress development-fold median is not positive and finite")

    return {
        "screen_pass": not reasons,
        "failure_reasons": reasons,
        "base": base,
        "stress": stress,
        "base_folds": base_folds,
        "stress_folds": stress_folds,
        "base_fold_median_return_pct": base_median,
        "stress_fold_median_return_pct": stress_median,
    }


def candidate_gate(size_reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Require the same screen result at exactly $4,000 and $6,000."""

    required = {"4000", "6000"}
    reasons: list[str] = []
    if set(size_reports) != required:
        reasons.append("screen requires exactly $4,000 and $6,000 reports")
    for size in sorted(required):
        report = size_reports.get(size)
        if not isinstance(report, Mapping) or report.get("screen_pass") is not True:
            reasons.append(f"{chr(36)}{size} does not pass the development screen")
    medians = [
        _number(report, "stress_fold_median_return_pct")
        for report in size_reports.values()
        if isinstance(report, Mapping)
    ]
    max_drawdown = max(
        (
            _number(result, "max_drawdown_pct")
            for report in size_reports.values()
            if isinstance(report, Mapping)
            for result in (report.get("base"), report.get("stress"))
            if isinstance(result, Mapping)
        ),
        default=math.nan,
    )
    robust_score = min(medians) - 0.25 * max_drawdown if (
        medians
        and all(math.isfinite(value) for value in medians)
        and math.isfinite(max_drawdown)
    ) else math.nan
    return {
        "screen_pass": not reasons,
        "failure_reasons": reasons,
        "minimum_stress_fold_median_return_pct": min(medians) if medians else math.nan,
        "maximum_development_drawdown_pct": max_drawdown,
        "robust_rank_score": robust_score,
        "eligible_for_promotion": False,
        "future_confirmation_required": True,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_exploration(
    *,
    btc_path: Path,
    eth_path: Path,
    output: Path,
    context_path: Path | None = None,
    max_data_age_days: float = DEFAULT_MAX_DATA_AGE_DAYS,
) -> dict[str, object]:
    """Run the fixed screen and save a research-only artifact."""

    pair, data_quality = validate_data(
        btc_path,
        eth_path,
        minimum_aligned_bars=DEFAULT_MIN_ALIGNED_BARS,
        max_data_age_days=max_data_age_days,
    )
    development_end, folds = development_folds(len(pair))
    context_events = load_context_events(context_path) if context_path else []
    candidates = candidate_definitions()
    results: list[dict[str, object]] = []

    for name, config in candidates.items():
        feature_map = build_pair_features(pair, config, context_events)
        sizes = {
            str(int(notional)): screen_size(
                pair,
                config,
                feature_map,
                order_notional=notional,
                development_end=development_end,
                folds=folds,
            )
            for notional in ORDER_NOTIONALS
        }
        gate = candidate_gate(sizes)
        results.append(
            {
                "candidate": name,
                "parameters": config.as_dict(),
                "sizes": sizes,
                "development_gate": gate,
            }
        )

    results.sort(
        key=lambda item: (
            not bool(item["development_gate"]["screen_pass"]),
            -float(item["development_gate"]["robust_rank_score"])
            if item["development_gate"]["robust_rank_score"] is not None
            else math.inf,
            str(item["candidate"]),
        )
    )
    shortlist = [
        str(item["candidate"])
        for item in results
        if item["development_gate"]["screen_pass"] is True
    ]
    report = {
        "schema_version": 1,
        "suite_version": SUITE_VERSION,
        "research_only": True,
        "strategy_ready": False,
        "active_profile_changed": False,
        "paper_orders_placed": False,
        "leverage_enabled": False,
        "automatic_promotion": False,
        "purpose": "development screening only",
        "data_quality": data_quality,
        "costs": {"base": BASE_COSTS, "stress": STRESS_COSTS},
        "required_order_notionals": list(ORDER_NOTIONALS),
        "context": {
            "path": str(context_path) if context_path else None,
            "events": len(context_events),
            "as_of_only": True,
            "neutral_when_missing": True,
        },
        "development_folds": [
            {"start_index": start, "end_index": end} for start, end in folds
        ],
        "protected_confirmation_holdout": {
            "start_index": development_end,
            "end_index": len(pair),
            "bars": len(pair) - development_end,
            "evaluated": False,
            "reason": "reserved for a later, separately frozen confirmation run",
        },
        "candidate_count": len(results),
        "shortlist_for_manual_freeze_only": shortlist,
        "candidates": results,
    }
    safe_report = _json_safe(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(safe_report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return safe_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-csv", type=Path)
    parser.add_argument(
        "--max-data-age-days",
        type=float,
        default=DEFAULT_MAX_DATA_AGE_DAYS,
    )
    args = parser.parse_args()
    report = run_exploration(
        btc_path=args.btc_path,
        eth_path=args.eth_path,
        output=args.output,
        context_path=args.context_csv,
        max_data_age_days=args.max_data_age_days,
    )
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "shortlist_for_manual_freeze_only": report[
                    "shortlist_for_manual_freeze_only"
                ],
                "strategy_ready": False,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
