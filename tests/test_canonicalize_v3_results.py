from __future__ import annotations

import json

from scripts.canonicalize_v3_results import canonicalize_state
from scripts.run_governed_research_cycle import iter_results, validate_result


def _candidate() -> dict[str, object]:
    result = {
        "return_pct": 1.0,
        "max_drawdown_pct": 2.0,
        "trades": 10,
        "entries": 10,
        "exits": 10,
    }
    return {
        "full_sample": result,
        "full_sample_stress": result,
        "promotion_gate": {"pass": False},
    }


def _report() -> dict[str, object]:
    return {
        "candidates": {
            "balanced": _candidate(),
            "selective": _candidate(),
            "conservative": _candidate(),
        }
    }


def test_canonicalize_state_uses_notional_and_strategy_candidate_dimensions(tmp_path):
    report_4000 = tmp_path / "report-4000.json"
    report_6000 = tmp_path / "report-6000.json"
    report_4000.write_text(json.dumps(_report()), encoding="utf-8")
    report_6000.write_text(json.dumps(_report()), encoding="utf-8")

    output = canonicalize_state(
        {
            "last_run": {
                "report_paths": {
                    "4000": str(report_4000),
                    "6000": str(report_6000),
                }
            }
        }
    )

    assert sorted(output) == [
        "4000:balanced",
        "4000:conservative",
        "4000:selective",
        "6000:balanced",
        "6000:conservative",
        "6000:selective",
    ]
    assert output["4000:balanced"]["candidate_id"] == "4000:balanced"
    assert output["6000:balanced"]["candidate_id"] == "6000:balanced"
    for candidate_id, result in iter_results({"candidates": output}):
        validate_result(result, candidate_id)
