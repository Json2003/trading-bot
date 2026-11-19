"""Manifest-fed monthly momentum comparison using an isolated cash ledger.

The fill model is an explicitly labelled daily-bar diagnostic. Source fetching,
model training, brokerage clients and promotion are outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean, median, stdev
from zoneinfo import ZoneInfo

from research.engine import research_safety_flags
from research.nyse_accounting import CashLedger, ZERO, decimal, positive, whole


class DataGap(ValueError):
    """An input cannot support the frozen calculation."""


def timestamp(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise DataGap(f"invalid availability timestamp {value!r}") from exc
    if stamp.tzinfo is None:
        raise DataGap("availability timestamps must include a UTC offset")
    return stamp


def known(record: dict, cutoff: str, label: str) -> None:
    if timestamp(record.get("available_at")) > timestamp(cutoff):
        raise DataGap(f"{label} was unavailable at {cutoff}")


def read_json(raw: bytes):
    def reject_constant(value):
        raise DataGap(f"invalid number {value}")

    def duplicate_checked(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise DataGap(f"duplicate JSON field {key}")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=duplicate_checked, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DataGap(f"invalid JSON input: {exc}") from exc


@dataclass
class Dataset:
    manifest: dict
    calendar: list[dict]
    securities: dict[str, dict]
    prices: dict[tuple[str, str], dict]
    actions: dict[str, list[dict]]
    fingerprints: dict[str, str]

    @property
    def sessions(self) -> list[str]:
        return [row["session"] for row in self.calendar]

    def price(self, security_id: str, session: str) -> dict:
        try:
            return self.prices[security_id, session]
        except KeyError as exc:
            raise DataGap(f"missing price for {security_id} on {session}") from exc


def validate_spec(spec: dict) -> None:
    if not isinstance(spec, dict):
        raise DataGap("specification must be a JSON object")
    required = {
        "experiment_id": "nyse_momentum_retention_v1", "research_only": True,
        "initial_cash": "5000", "max_position_notional": "1000", "cost_bps_per_side": "43",
        "top_k": 5, "retain_rank": 10, "lookback_sessions": 252, "skip_sessions": 21,
        "liquidity_sessions": 20, "minimum_price": "5", "minimum_median_dollar_volume": "20000000",
        "risk_fraction": "0.005", "stop_atr_multiple": "2", "drawdown_latch_fraction": "0.02",
        "prior_volume_participation": "0.001", "minimum_independent_observations": 20,
        "signal_price_basis": "asof_total_return_index", "promotion_allowed": False,
    }
    for key, expected in required.items():
        if type(spec.get(key)) is not type(expected) or spec[key] != expected:
            raise DataGap(f"frozen specification field {key} must equal {expected!r}")
    if set(spec) - required.keys() - {"schema_version", "notes"}:
        raise DataGap("unrecognized frozen specification fields")


def load_dataset(manifest_path: Path, spec: dict, *, manifest_raw: bytes | None = None) -> Dataset:
    manifest = read_json(manifest_path.read_bytes() if manifest_raw is None else manifest_raw)
    if not isinstance(manifest, dict):
        raise DataGap("manifest must be a JSON object")
    if manifest.get("schema_version") != 1 or manifest.get("data_kind") not in {"market", "synthetic"}:
        raise DataGap("manifest requires schema_version 1 and market or synthetic data_kind")
    if manifest.get("evidence_role") != "development_diagnostic":
        raise DataGap("this runner supports development diagnostics only")
    if manifest.get("signal_price_basis") != spec["signal_price_basis"]:
        raise DataGap("signal price basis must be an as-of total-return index")
    if not manifest.get("dataset_id") or not manifest.get("universe_selection"):
        raise DataGap("dataset identity and prior universe selection rationale are required")
    blocks = manifest.get("blocks", [])
    if not blocks:
        raise DataGap("manifest must freeze evaluation blocks")
    inputs, fingerprints = {}, {}
    for name in ("calendar", "securities", "prices", "actions"):
        descriptor = manifest.get("files", {}).get(name, {})
        if not all(descriptor.get(k) for k in ("path", "sha256", "source", "revision", "received_at")):
            raise DataGap(f"{name} needs path, sha256, source, revision and received_at")
        receipt = timestamp(descriptor["received_at"])
        if manifest["data_kind"] == "market" and receipt > datetime.now(timezone.utc):
            raise DataGap(f"{name} receipt is in the future")
        raw = (manifest_path.parent / descriptor["path"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if descriptor["sha256"] != digest:
            raise DataGap(f"{name} SHA-256 mismatch")
        fingerprints[name] = digest
        inputs[name] = read_json(raw)
        if not isinstance(inputs[name], list):
            raise DataGap(f"{name} must be a JSON array")
    calendar = inputs["calendar"]
    dates = [row["session"] for row in calendar]
    if not dates or dates != sorted(set(dates)):
        raise DataGap("calendar needs sorted unique sessions")
    for row in calendar:
        if date.fromisoformat(row["session"]).weekday() > 4:
            raise DataGap("NYSE calendar contains a weekend session")
        if not timestamp(row["open_at"]) < timestamp(row["close_at"]) <= timestamp(row["decision_at"]):
            raise DataGap("session close must follow its open")
        opening = timestamp(row["open_at"]).astimezone(ZoneInfo("America/New_York"))
        if (opening.hour, opening.minute, opening.second) != (9, 30, 0):
            raise DataGap("calendar needs NYSE core opens at 09:30 New York time")
        if opening.date().isoformat() != row["session"]:
            raise DataGap("calendar date and open timestamp disagree")
    for prior, current in zip(calendar, calendar[1:]):
        if timestamp(prior["decision_at"]) >= timestamp(current["open_at"]):
            raise DataGap("decision must precede the following session open")
    names = set()
    previous_end = None
    for block in blocks:
        start, end = block["start"], block["end"]
        if block["name"] in names or (previous_end and start <= previous_end) or start > end:
            raise DataGap("blocks must have unique names and nonoverlapping chronological dates")
        names.add(block["name"])
        if (start not in dates or end not in dates
                or dates.index(start) < spec["lookback_sessions"] + 1
                or dates.index(end) + 2 >= len(dates)):
            raise DataGap("blocks need 253 prior observations and two following settlement sessions")
        previous_end = end
        for field in ("corporate_actions_complete_through", "membership_complete_through"):
            if manifest.get(field, "") < end:
                raise DataGap(f"{field} does not cover block {block['name']}")
    securities = {}
    for row in inputs["securities"]:
        sid = row["security_id"]
        if not isinstance(sid, str) or not sid.strip() or sid in securities:
            raise DataGap("permanent security IDs must be nonempty and unique")
        if not row.get("ticker") or row.get("currency") != "USD":
            raise DataGap(f"security {sid} needs a ticker and USD currency")
        timestamp(row["available_at"])
        date.fromisoformat(row["eligible_from"])
        if row.get("eligible_until") is not None:
            date.fromisoformat(row["eligible_until"])
            if row["eligible_until"] <= row["eligible_from"]:
                raise DataGap(f"empty membership interval for {sid}")
        securities[sid] = row
    cohort = manifest.get("universe", [])
    if len(cohort) != len(set(cohort)) or not cohort or set(cohort) - securities.keys():
        raise DataGap("frozen universe needs unique IDs present in securities")
    prices = {}
    for row in inputs["prices"]:
        key = row["security_id"], row["session"]
        if key in prices or key[0] not in securities or key[1] not in dates:
            raise DataGap(f"duplicate or unidentified price {key}")
        values = {k: positive(row[k], f"{key} {k}") for k in ("open", "high", "low", "close", "momentum_close")}
        if row.get("atr14") is not None:
            positive(row["atr14"], f"{key} atr14")
        if not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
            raise DataGap(f"inconsistent OHLC {key}")
        volume = positive(row["volume"], f"{key} volume")
        if volume != whole(volume):
            raise DataGap(f"noninteger volume {key}")
        availability = timestamp(row["available_at"])
        if availability < timestamp(calendar[dates.index(key[1])]["close_at"]):
            raise DataGap(f"completed OHLC cannot precede its session close {key}")
        if availability > timestamp(manifest["files"]["prices"]["received_at"]):
            raise DataGap(f"price availability is later than the file receipt {key}")
        prices[key] = row
    actions = {}
    ids = set()
    supported = {"dividend", "split", "stock_merger", "cash_merger", "rename", "fraction_payment"}
    for row in inputs["actions"]:
        action_id = row["action_id"]
        session = row["session"]
        if not isinstance(action_id, str) or not action_id.strip() or action_id in ids or session not in dates or row["security_id"] not in securities:
            raise DataGap("corporate actions need unique IDs, known securities and calendar sessions")
        if row["kind"] not in supported:
            raise DataGap("unsupported corporate action")
        if row.get("successor_id", row["security_id"]) not in securities:
            raise DataGap("merger successor is missing from security metadata")
        if row["kind"] in {"dividend", "cash_merger"}:
            if decimal(row["amount"]) < 0 or (row["kind"] == "cash_merger" and decimal(row["amount"]) == 0):
                raise DataGap("invalid corporate-action cash amount")
        if row["kind"] in {"split", "stock_merger"}:
            positive(row["new_shares"], "new share ratio")
            positive(row["old_shares"], "old share ratio")
        if row["kind"] == "fraction_payment":
            positive(row["price"], "fraction payment price")
            if not row.get("fraction_id"):
                raise DataGap("fraction payment needs an entitlement ID")
        if row["kind"] == "rename" and not all(row.get(k) for k in ("old_ticker", "new_ticker")):
            raise DataGap("ticker changes need both ticker spellings")
        if row.get("payable_date") is not None and date.fromisoformat(row["payable_date"]) < date.fromisoformat(session):
            raise DataGap("payment cannot precede the entitlement session")
        known(row, calendar[dates.index(session)]["open_at"], f"action {action_id}")
        ids.add(action_id)
        actions.setdefault(session, []).append(row)
    # File order is the explicit precedence for simultaneous actions.
    data = Dataset(manifest, calendar, securities, prices, actions, fingerprints)
    for block in blocks:
        validate_block_coverage(data, spec, block)
    return data


def entry_exclusion(security: dict, day: str) -> str | None:
    if not (security["eligible_from"] <= day
            and (security.get("eligible_until") is None or day < security["eligible_until"])):
        return "outside_membership_interval"
    if security.get("exchange") in {None, "", "UNKNOWN"} or security.get("instrument_type") in {None, "", "UNKNOWN"}:
        raise DataGap(f"unresolved exchange or instrument class for {security['security_id']}")
    if security["exchange"] != "XNYS" or security["instrument_type"] != "COMMON_STOCK":
        return "outside_nyse_common_stock_mandate"
    return None


def validate_block_coverage(data: Dataset, spec: dict, block: dict) -> None:
    """Require completed observation coverage even when a portfolio stays in cash."""
    sessions = data.sessions
    start, end = sessions.index(block["start"]), sessions.index(block["end"])
    price_receipt = timestamp(data.manifest["files"]["prices"]["received_at"])
    if timestamp(data.calendar[end]["close_at"]) > price_receipt:
        raise DataGap(f"block {block['name']} has no completed price receipt through {block['end']}")
    exits = {}
    for session, actions in data.actions.items():
        for action in actions:
            if action["kind"] in {"cash_merger", "stock_merger"}:
                sid = action["security_id"]
                exits[sid] = min(exits.get(sid, session), session)
    for i in range(start - 1, end + 1):
        row = data.calendar[i]
        day, cutoff = row["session"], row["decision_at"]
        month_end = day[:7] != sessions[i + 1][:7]
        if i < start and not month_end:
            continue
        for sid in data.manifest["universe"]:
            security = data.securities[sid]
            known(security, cutoff, f"metadata {sid}")
            if (sid in exits and day >= exits[sid]) or entry_exclusion(security, day):
                continue
            first = i - spec["lookback_sessions"] if month_end else i
            for session in sessions[first:i + 1]:
                known(data.price(sid, session), cutoff, f"price {sid}/{session}")
            positive(data.price(sid, day).get("atr14"), f"ATR for {sid}/{day}")


def rank_stocks(data: Dataset, decision_index: int, spec: dict, *, audit: list | None = None) -> list[dict]:
    """Every eligible candidate is ranked or produces an explicit input failure."""
    calendar = data.calendar
    current = calendar[decision_index]
    cutoff = current["decision_at"]
    terminated = {row["security_id"] for session, rows in data.actions.items()
                  if session <= current["session"] for row in rows
                  if row["kind"] in {"cash_merger", "stock_merger"}}
    ranked = []
    for sid in data.manifest["universe"]:
        security = data.securities[sid]
        known(security, cutoff, f"metadata {sid}")
        if sid in terminated:
            if audit is not None:
                audit.append({"security_id": sid, "eligible": False, "reason": "dated_merger_exit"})
            continue
        day = current["session"]
        exclusion = entry_exclusion(security, day)
        if exclusion:
            if audit is not None:
                audit.append({"security_id": sid, "eligible": False, "reason": exclusion})
            continue
        needed = calendar[decision_index - spec["lookback_sessions"]:decision_index + 1]
        if len(needed) != 253:
            raise DataGap("momentum needs the observation at a 252-session lag")
        history = [data.price(sid, row["session"]) for row in needed]
        for row in history:
            known(row, cutoff, f"price {sid}/{row['session']}")
        latest = history[-1]
        dollar_volume = median(decimal(p["close"]) * decimal(p["volume"]) for p in history[-spec["liquidity_sessions"]:])
        if decimal(latest["close"]) <= decimal(spec["minimum_price"]) or dollar_volume < decimal(spec["minimum_median_dollar_volume"]):
            if audit is not None:
                audit.append({"security_id": sid, "eligible": False, "reason": "price_or_liquidity_threshold"})
            continue
        score = decimal(history[-1 - spec["skip_sessions"]]["momentum_close"]) / decimal(history[0]["momentum_close"]) - 1
        ranked.append({"security_id": sid, "score": str(score)})
        if audit is not None:
            audit.append({"security_id": sid, "eligible": True, "score": str(score)})
    ranked.sort(key=lambda r: (-decimal(r["score"]), r["security_id"]))
    return ranked


def select_targets(ranking: list[dict], held: set[str], buffered: bool, spec: dict) -> list[str]:
    order = [row["security_id"] for row in ranking]
    retained = [sid for sid in order[:spec["retain_rank"]] if sid in held] if buffered else []
    for sid in order:
        if len(retained) >= spec["top_k"]:
            break
        if sid not in retained:
            retained.append(sid)
    return retained[:spec["top_k"]]


def independent_count(realizations: list[dict]) -> int:
    """Reuse PR #290's nonoverlapping-interval safeguard."""
    count, last_exit = 0, ""
    for row in sorted(realizations, key=lambda t: (t["exit_session"], t["entry_session"])):
        if row["entry_session"] > last_exit:
            count += 1
            last_exit = row["exit_session"]
    return count


def metrics(ledger: CashLedger, curve: list[dict], initial: Decimal, minimum: int, decisions: list[dict]) -> dict:
    values = [initial] + [decimal(row["equity"]) for row in curve]
    changes = [float(b / a - 1) for a, b in zip(values, values[1:])]
    high, drawdown = initial, ZERO
    for value in values:
        high = max(high, value)
        drawdown = min(drawdown, value / high - 1)
    profits = [decimal(row["net_pnl"]) for row in ledger.realizations]
    wins = sum((p for p in profits if p > 0), ZERO)
    losses = -sum((p for p in profits if p < 0), ZERO)
    observations = independent_count(ledger.realizations)
    decision_dates = [row["session"] for row in decisions]
    closed_cohorts = {max(day for day in decision_dates if day < row["entry_session"])
                      for row in ledger.realizations}
    independent_observations = min(observations, len(closed_cohorts))
    deviation = stdev(changes) if len(changes) > 1 else 0
    result = {"net_return": float(values[-1] / initial - 1), "drawdown": float(drawdown),
            "sharpe": mean(changes) / deviation * math.sqrt(252) if deviation else None,
            "profit_factor": float(wins / losses) if losses else None,
            "trade_count": sum(row["kind"] == "sell" for row in ledger.events),
            "fill_count": sum(row["kind"] in {"buy", "sell"} for row in ledger.events),
            "execution_costs": float(ledger.costs), "turnover": float(ledger.traded_notional / initial),
            "nonoverlapping_observations": observations,
            "closed_monthly_cohort_dates": sorted(closed_cohorts),
            "independent_observations": independent_observations,
            "minimum_sample_met": independent_observations >= minimum,
            "metric_units": {"net_return": "fraction", "drawdown": "negative_fraction", "execution_costs": "USD",
                             "turnover": "traded_notional_over_initial_cash", "sharpe": "annualized_daily_zero_rf"}}
    for key in ("net_return", "drawdown", "sharpe", "profit_factor", "execution_costs", "turnover"):
        if result[key] is not None and not math.isfinite(result[key]):
            raise DataGap(f"metric {key} is nonfinite")
    return result


def replay(data: Dataset, spec: dict, block: dict, buffered: bool) -> dict:
    ledger = CashLedger(data.sessions, spec["initial_cash"], spec["max_position_notional"], spec["cost_bps_per_side"])
    curve, decisions, intentions = [], [], []
    targets: list[str] = []
    stopped: set[str] = set()
    pending: list[dict] = []
    latched = False
    latch_at = None
    peak = decimal(spec["initial_cash"])
    indices = [i for i, row in enumerate(data.calendar) if block["start"] <= row["session"] <= block["end"]]
    # The first open may execute a decision made on the preceding month's close.
    for i in range(indices[0] - 1, indices[-1] + 1):
        day = data.sessions[i]
        if i >= indices[0]:
            ledger.begin(day)
            transformed = set()
            for action in data.actions.get(day, []):
                ledger.apply_action(action)
                if action["kind"] in {"split", "stock_merger", "cash_merger"}:
                    transformed.add(action["security_id"])
                if action["kind"] in {"stock_merger", "cash_merger"}:
                    # Mandatory shares may leave the NYSE mandate and are queued for exit below.
                    targets = [s for s in targets if s != action["security_id"]]
            targets = [s for s in targets if not entry_exclusion(data.securities[s], day)]
            for order in pending:
                sid = order["security_id"]
                if sid in transformed:
                    ledger.log("rejected_order", order=order, reason="corporate_action_requires_fresh_quantity")
                    continue
                if order["side"] == "buy" and entry_exclusion(data.securities[sid], day):
                    ledger.log("rejected_order", order=order, reason="security_ineligible_at_open")
                    continue
                bar = data.price(sid, day)
                known(bar, data.calendar[i]["decision_at"], f"execution bar {sid}/{day}")
                opening = positive(bar["open"])
                if order["side"] == "sell":
                    pos = ledger.positions.get(sid)
                    if pos:
                        ledger.sell(sid, min(order["quantity"], pos.quantity), opening, order["reason"])
                elif sid not in ledger.positions and not latched:
                    notional = opening * order["quantity"]
                    if notional > ledger.cap or notional * (1 + ledger.cost_rate) > ledger.cash:
                        ledger.log("rejected_order", order=order, reason="opening_gap_exceeds_cash_or_cap")
                    elif opening <= decimal(order["stop_distance"]):
                        ledger.log("rejected_order", order=order, reason="nonpositive_stop")
                    else:
                        ledger.buy(sid, order["quantity"], opening, opening - decimal(order["stop_distance"]))
            for sid, pos in list(ledger.positions.items()):
                bar = data.price(sid, day)
                known(bar, data.calendar[i]["decision_at"], f"holding bar {sid}/{day}")
                if positive(bar["low"]) <= pos.stop:
                    ledger.sell(sid, pos.quantity, min(positive(bar["open"]), pos.stop), "adverse_daily_stop_proxy")
                    stopped.add(sid)
            marks = {sid: row["close"] for (sid, session), row in data.prices.items() if session == day}
            state = ledger.snapshot(marks)
            equity = decimal(state["equity"])
            peak = max(peak, equity)
            if not latched and equity / peak - 1 <= -decimal(spec["drawdown_latch_fraction"]):
                latched, latch_at = True, day
            state["drawdown_latched"] = latched
            curve.append(state)
        else:
            ledger.session = day
            equity = decimal(spec["initial_cash"])
        month_end = data.sessions[i][:7] != data.sessions[i + 1][:7]
        if month_end:
            eligibility = []
            ranking = rank_stocks(data, i, spec, audit=eligibility)
            targets = select_targets(ranking, set(ledger.positions), buffered, spec)
            stopped.clear()
            decisions.append({"session": day, "available_at": data.calendar[i]["decision_at"],
                              "ranking": ranking, "eligibility": eligibility, "target_ids": list(targets)})
        pending = []
        for sid, pos in ledger.positions.items():
            current = data.price(sid, day)
            known(current, data.calendar[i]["decision_at"], f"held price {sid}/{day}")
            price = positive(current["close"])
            quantity = pos.quantity if latched or sid not in targets else max(0, pos.quantity - whole(ledger.cap / price))
            if quantity:
                pending.append({"side": "sell", "security_id": sid, "quantity": quantity,
                                "reason": "drawdown_latch" if latched else "target_or_cap"})
        reserved = ZERO
        if not latched:
            for sid in targets:
                if sid in ledger.positions or sid in stopped:
                    continue
                current = data.price(sid, day)
                known(current, data.calendar[i]["decision_at"], f"entry price {sid}/{day}")
                price, distance = positive(current["close"]), positive(current["atr14"]) * decimal(spec["stop_atr_multiple"])
                budget = min(ledger.cap, max(ZERO, ledger.cash - reserved) / (1 + ledger.cost_rate))
                quantity = min(whole(budget / price), whole(equity * decimal(spec["risk_fraction"]) / distance),
                               whole(decimal(current["volume"]) * decimal(spec["prior_volume_participation"])))
                if quantity:
                    pending.append({"side": "buy", "security_id": sid, "quantity": quantity,
                                    "stop_distance": str(distance), "reason": "frozen_monthly_target"})
                    reserved += price * quantity * (1 + ledger.cost_rate)
        if i < indices[-1]:
            intentions.append({"decision_session": day, "decision_at": data.calendar[i]["decision_at"],
                               "execution_session": data.sessions[i + 1], "orders": pending})
    result = metrics(ledger, curve, decimal(spec["initial_cash"]), spec["minimum_independent_observations"], decisions)
    return {"block": block["name"], "arm": "buffered" if buffered else "top_five", **result,
            "drawdown_latched_at": latch_at, "decisions": decisions, "intentions": intentions,
            "ledger": ledger.events, "realizations": ledger.realizations, "equity_curve": curve,
            "observation_ids": [f"{spec['experiment_id']}:{day}" for day in result["closed_monthly_cohort_dates"]],
            "undated_receivables": [k for k, v in ledger.receivables.items() if v.due is None]}


def evaluate(data: Dataset, spec: dict) -> dict:
    results = []
    comparisons = []
    for block in data.manifest["blocks"]:
        pair = [replay(data, spec, block, arm) for arm in (False, True)]
        if pair[0]["decisions"] and [d["ranking"] for d in pair[0]["decisions"]] != [d["ranking"] for d in pair[1]["decisions"]]:
            raise ValueError("comparison arms used different rankings")
        for row in pair:
            row["failure_reasons"] = []
            if not row["minimum_sample_met"]:
                row["failure_reasons"].append("insufficient_independent_observations")
            if row["drawdown_latched_at"] is not None:
                row["failure_reasons"].append("drawdown_latch_triggered")
            if row["net_return"] <= 0:
                row["failure_reasons"].append("nonpositive_net_return")
            if row["undated_receivables"]:
                row["failure_reasons"].append("cash_receipt_dates_unverified")
            row["unavailable_metrics"] = [k for k in ("sharpe", "profit_factor") if row[k] is None]
        comparisons.append({"block": block["name"], "net_return_difference": pair[1]["net_return"] - pair[0]["net_return"],
                            "cost_savings_usd": pair[0]["execution_costs"] - pair[1]["execution_costs"],
                            "turnover_reduction": pair[0]["turnover"] - pair[1]["turnover"]})
        results.extend(pair)
    synthetic = data.manifest["data_kind"] == "synthetic"
    return {"schema_version": 1, **research_safety_flags(),
            "status": "software_fixture_complete" if synthetic else "historical_diagnostic_complete",
            "dataset_id": data.manifest["dataset_id"], "data_kind": data.manifest["data_kind"],
            "evidence_role": "development_diagnostic", "experiment_id": spec["experiment_id"],
            "fill_model": "previous_close_whole_share_intentions_next_raw_open_and_adverse_daily_stop_proxy",
            "results": results, "comparisons": comparisons,
            "checkpoint_advanced": False, "forward_confirmation": False,
            "investment_trials": 0 if synthetic else 1, "model_training_runs": 0,
            "eligible_for_confirmation": False,
            "failure_reasons": ["order_time_quotes_and_size_unverified", "development_diagnostic_only"],
            "limitations": ["Daily bar stops assume an adverse executable price and sufficient size.",
                            "Receipts use supplied payable dates, with undated claims kept unavailable.",
                            "Nonoverlapping holding intervals are a sample safeguard and may remain correlated.",
                            "Input coverage and as-of feature declarations require independent source audit."]}
