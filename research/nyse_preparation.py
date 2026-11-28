"""Prepare raw-dollar bars, causal total returns and manifest files from receipts."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from research.nyse_accounting import ZERO, decimal, positive
from research.nyse_capture import DOLT_API, calendar, captured_bars, digest, encode, save_new
from research.nyse_offline import DataGap, entry_exclusion, load_dataset, read_json, timestamp, validate_spec


def enrich_prices(bars: dict, securities: list[dict], sessions: list[dict], actions: list[dict]) -> list[dict]:
    by_session = defaultdict(list)
    for action in actions:
        by_session[action["security_id"], action["session"]].append(action)
    prices = []
    for security in securities:
        sid, ticker = security["security_id"], security["ticker"]
        previous_close = average = None
        tr_seed = []
        total_return = Decimal(100)
        available = timestamp(security["available_at"])
        for row in sessions:
            day = row["session"]
            if (ticker, day) not in bars:
                # Do not bridge a missing session with indicators from a broken series.
                break
            bar = bars[ticker, day]
            available = max(available, timestamp(bar["available_at"]))
            ratio, dividend = Decimal(1), ZERO
            for action in by_session[sid, day]:
                available = max(available, timestamp(action["available_at"]))
                if action["kind"] == "split":
                    ratio *= positive(action["new_shares"]) / positive(action["old_shares"])
                elif action["kind"] == "dividend":
                    amount = decimal(action["amount"])
                    if amount < 0:
                        raise DataGap(f"negative dividend for {ticker}/{day}")
                    dividend += amount
                else:
                    raise DataGap(f"capture adapter needs review for {action['kind']} on {ticker}/{day}")
            close, high, low = (decimal(bar[k]) for k in ("close", "high", "low"))
            if previous_close is not None:
                reference = previous_close / ratio
                total_return *= (close + dividend) / reference
                tr = max(high - low, abs(high - reference), abs(low - reference))
                if average is None:
                    tr_seed = [v / ratio for v in tr_seed] + [tr]
                    if len(tr_seed) == 14:
                        average = sum(tr_seed) / 14
                else:
                    average = (13 * average / ratio + tr) / 14
            prices.append({**bar, "security_id": sid, "session": day,
                           "available_at": available.isoformat(), "momentum_close": str(total_return),
                           "atr14": str(average) if average else None})
            previous_close = close
    return prices


def prepare(plan: dict, plan_hash: str, captures: Path, review_path: Path, output: Path, spec: dict) -> dict:
    validate_spec(spec)
    review_raw = review_path.read_bytes()
    review = read_json(review_raw)
    if review.get("schema_version") != 1 or review.get("plan_sha256") != plan_hash:
        raise DataGap("review must reference the frozen capture plan")
    now = datetime.now(timezone.utc)
    reviewed_at = timestamp(review["reviewed_at"])
    if reviewed_at > now or not review.get("evidence"):
        raise DataGap("review needs observed time and primary-source evidence")
    securities = review["securities"]
    if (sorted(s["ticker"] for s in securities) != sorted(plan["symbols"])
            or len({s["security_id"] for s in securities}) != len(securities)):
        raise DataGap("security review must reconcile every frozen ticker to a distinct class")
    block = plan["block"]
    sessions = calendar(plan)
    dates = [r["session"] for r in sessions]
    first_decision = dates.index(block["start"]) - 1
    previous = sessions[first_decision]
    if previous["session"][:7] == block["start"][:7] or first_decision < spec["lookback_sessions"]:
        raise DataGap("plan needs a preceding month-end and a complete initial lookback")
    if any(timestamp(s["available_at"]) > reviewed_at for s in securities):
        raise DataGap("security record appears after its review time")
    excluded = {r["ticker"]: reason for r in securities if (reason := entry_exclusion(r, previous["session"]))}
    active = [r for r in securities if r["ticker"] not in excluded]
    if not active:
        raise DataGap("plan has no eligible NYSE common stocks at the first decision")
    if review.get("price_basis") != "nominal_ohlc" or not review.get("price_basis_evidence"):
        raise DataGap("review must establish nominal OHLC conventions and record evidence")
    # This preparation path expects stable eligible identities. More complex actions remain in the offline ledger.
    advisory_actions = review["actions"]
    operating = [r for r in sessions if r["session"] <= block["end"]]
    id_set = {r["security_id"] for r in active}
    open_times = {r["session"]: timestamp(r["open_at"]) for r in operating}
    action_ids = set()
    for action in advisory_actions:
        if (action["security_id"] not in id_set or action["session"] not in open_times
                or action["action_id"] in action_ids or action["kind"] not in {"dividend", "split"}
                or not action.get("source")):
            raise DataGap("unreconciled action for the frozen preparation path")
        received = timestamp(action["available_at"])
        if received > reviewed_at:
            raise DataGap("action appears after its review time")
        if action["session"] >= block["start"] and received > open_times[action["session"]]:
            raise DataGap(f"action {action['action_id']} was received after its affected open")
        action_ids.add(action["action_id"])
    bars, provenance = captured_bars(plan, plan_hash, captures)
    symbols = {r["ticker"] for r in active}
    completed = [r for r in operating if timestamp(r["close_at"]) <= now]
    complete_sessions = []
    for row in completed:
        if any((ticker, row["session"]) not in bars for ticker in symbols):
            break
        complete_sessions.append(row)
    observed = complete_sessions[-1]["session"] if complete_sessions else None
    required = [(ticker, row["session"]) for row in completed for ticker in sorted(symbols)]
    missing = [key for key in required if key not in bars]
    covered = review.get("corporate_actions_complete_through")
    membership = review.get("membership_complete_through")
    if any(value and (value not in open_times or timestamp(operating[dates.index(value)]["close_at"]) > reviewed_at)
           for value in (covered, membership)):
        raise DataGap("coverage review claims a session not completed by the review time")
    signal_rows = []
    if observed and covered:
        signal_rows = enrich_prices(bars, active, [r for r in complete_sessions if r["session"] <= covered], advisory_actions)
    issues = []
    if missing:
        issues.append("completed_session_bars_missing")
    if not observed or observed < block["end"]:
        issues.append("forward_window_incomplete")
    if not covered or covered < block["end"]:
        issues.append("corporate_action_review_incomplete")
    if not membership or membership < block["end"]:
        issues.append("membership_review_incomplete")
    report = {"status": "awaiting_market_data" if issues else "ready_for_offline_replay",
              "dataset_id": plan["dataset_id"], "plan_sha256": plan_hash, "review_sha256": digest(review_raw),
              "expected_block": block, "first_decision_at": previous["decision_at"],
              "eligible_tickers_at_start": sorted(symbols), "excluded_tickers": excluded,
              "captured_bars": len(bars), "eligible_bars_checked": len(required) - len(missing),
              "last_contiguous_session": observed, "completed_sessions_checked": len(complete_sessions),
              "prepared_signal_rows": len(signal_rows), "missing_count": len(missing),
              "first_missing": missing[:12], "action_review_through": covered, "membership_review_through": membership,
              "issues": issues, "capture_provenance": provenance, "strategy_trials": 0, "expected_profit": None}
    if issues:
        return report
    # The capture adapter carries stable classes. A new exchange/ticker change within the window
    # needs a reviewed manifest for the general offline runner.
    receipt_time = max([reviewed_at] + [timestamp(r["available_at"]) for r in signal_rows]).isoformat()
    inputs = {"calendar": sessions, "securities": securities, "prices": signal_rows,
              "actions": [a for a in advisory_actions if a["session"] >= block["start"]]}
    manifest = {"schema_version": 1, "dataset_id": plan["dataset_id"], "data_kind": "market",
                "evidence_role": "development_diagnostic", "signal_price_basis": spec["signal_price_basis"],
                "universe_selection": plan["universe_selection"], "universe": [r["security_id"] for r in securities],
                "blocks": [block], "corporate_actions_complete_through": covered,
                "membership_complete_through": membership,
                "source_evidence": {"capture_plan_sha256": plan_hash,
                                    "input_review_sha256": digest(review_raw),
                                    "capture_receipts": provenance},
                "files": {}}
    source_names = {"calendar": " | ".join(plan["calendar_sources"]), "prices": DOLT_API,
                    "securities": "SEC and reviewed listing records", "actions": "Reviewed issuer and ex-date records"}
    for name, records in inputs.items():
        data = encode(records)
        sha = digest(data)
        relative = "files/" + name + "-" + sha + ".json"
        save_new(output / relative, data)
        manifest["files"][name] = {"path": relative, "sha256": sha, "source": source_names[name],
                                   "revision": plan_hash if name == "calendar" else sha,
                                   "received_at": receipt_time}
    raw_manifest = encode(manifest)
    path = output / ("manifest-" + digest(raw_manifest) + ".json")
    load_dataset(path, spec, manifest_raw=raw_manifest)
    save_new(path, raw_manifest)
    report["manifest"] = str(path)
    return report
