"""Exercise the actual offline entry point with clearly synthetic market inputs."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from research.nyse_offline import DataGap, evaluate, load_dataset, rank_stocks, select_targets

ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location("offline_runner", ROOT / "scripts/run_nyse_offline_research.py")
runner = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(runner)
SPEC_PATH = ROOT / "research/experiments/nyse_momentum_retention_v1.json"


def make_inputs(directory: Path) -> tuple[dict, dict, dict]:
    """Frozen software examples with engineered rankings and constant raw prices."""
    ny = ZoneInfo("America/New_York")
    sessions = []
    day = date(2024, 10, 1)
    while day <= date(2026, 2, 10):
        if day.weekday() < 5 and day.isoformat() not in {"2025-12-25", "2026-01-01", "2026-01-19"}:
            sessions.append({"session": day.isoformat(),
                             "open_at": datetime.combine(day, time(9, 30), ny).isoformat(),
                             "close_at": datetime.combine(day, time(16), ny).isoformat(),
                             "decision_at": datetime.combine(day, time(16, 5), ny).isoformat()})
        day += timedelta(days=1)
    ids = [f"SEC{i:02d}" for i in range(1, 13)]
    securities = [{"security_id": sid, "ticker": sid, "exchange": "XNYS", "instrument_type": "COMMON_STOCK",
                   "currency": "USD", "eligible_from": sessions[0]["session"], "eligible_until": None,
                   "available_at": "2020-01-01T00:00:00Z"} for sid in ids]
    prices = []
    for index, session in enumerate(sessions):
        for rank, sid in enumerate(ids):
            prices.append({"security_id": sid, "session": session["session"], "available_at": session["close_at"],
                           "open": "100", "high": "101", "low": "99", "close": "100", "volume": 1000000,
                           "momentum_close": str(Decimal(100) + Decimal(index * (13 - rank)) / 100), "atr14": "1"})
    index = next(i for i, row in enumerate(sessions) if row["session"] == "2026-01-30")
    lookup = {(p["security_id"], p["session"]): p for p in prices}
    desired = ["SEC02", "SEC03", "SEC04", "SEC05", "SEC06", "SEC01"] + ids[6:]
    for rank, sid in enumerate(desired):
        denominator = lookup[sid, sessions[index - 252]["session"]]["momentum_close"]
        lookup[sid, sessions[index - 21]["session"]]["momentum_close"] = str(Decimal(denominator) * (1 + Decimal(20 - rank) / 100))
    manifest = {"schema_version": 1, "dataset_id": "synthetic-rank-retention-software-fixture",
                "data_kind": "synthetic", "evidence_role": "development_diagnostic",
                "signal_price_basis": "asof_total_return_index", "universe": ids,
                "universe_selection": "Twelve synthetic IDs fixed before tests. Engineered ranks are software examples.",
                "blocks": [{"name": "software", "start": "2026-01-02", "end": "2026-02-05"}],
                "corporate_actions_complete_through": "2026-02-10", "membership_complete_through": "2026-02-10",
                "files": {}}
    inputs = {"calendar": sessions, "securities": securities, "prices": prices, "actions": []}
    write_inputs(directory, manifest, inputs)
    return manifest, inputs, json.loads(SPEC_PATH.read_text())


def write_inputs(directory: Path, manifest: dict, inputs: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in inputs.items():
        raw = json.dumps(value, sort_keys=True).encode()
        (directory / f"{name}.json").write_bytes(raw)
        manifest["files"][name] = {"path": f"{name}.json", "sha256": hashlib.sha256(raw).hexdigest(),
                                    "source": "first-party synthetic test fixture", "revision": "test-v1",
                                    "received_at": "2026-11-18T00:00:00Z"}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


class OfflineResearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.manifest, self.inputs, self.spec = make_inputs(self.directory)
        self.manifest_path = self.directory / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def data(self):
        write_inputs(self.directory, self.manifest, self.inputs)
        return load_dataset(self.manifest_path, self.spec)

    def test_buffer_retains_rank_six_and_naive_replaces_it(self):
        report = evaluate(self.data(), self.spec)
        naive, buffered = report["results"]
        self.assertEqual(naive["decisions"][0]["target_ids"], self.manifest["universe"][:5])
        self.assertEqual(naive["decisions"][1]["target_ids"], ["SEC02", "SEC03", "SEC04", "SEC05", "SEC06"])
        self.assertEqual(buffered["decisions"][1]["target_ids"], ["SEC02", "SEC03", "SEC04", "SEC05", "SEC01"])
        self.assertGreater(naive["execution_costs"], buffered["execution_costs"])
        for arm in (naive, buffered):
            pnl = {key: Decimal(value) for key, value in arm["pnl_usd"].items()}
            fees = sum(Decimal(row["fee"]) for row in arm["ledger"] if row["kind"] in {"buy", "sell"})
            self.assertEqual(pnl["stress_expenses"], fees)
            self.assertEqual(pnl["net_marked"], Decimal(arm["equity_curve"][-1]["equity"]) - Decimal("5000"))
            self.assertEqual(pnl["net_realized"], sum(Decimal(row["net_pnl"]) for row in arm["realizations"]))
            self.assertEqual(pnl["net_realized"] + pnl["net_unrealized"], pnl["net_marked"])
            # Flat prices with no actions isolate the impact of the modeled fees.
            self.assertEqual(pnl["gross_marked"], Decimal("0"))
            self.assertEqual(pnl["net_marked"], -fees)
        self.assertEqual(Decimal(report["comparisons"][0]["gross_marked_pnl_difference_usd"]), 0)
        # A February 2 sale cannot finance its replacement until settlement.
        sec6 = [r for r in naive["ledger"] if r["kind"] == "buy" and r["security_id"] == "SEC06"]
        self.assertEqual(sec6[-1]["session"], "2026-02-04")
        for arm in report["results"]:
            for row in arm["equity_curve"]:
                total = sum(Decimal(row[k]) for k in ("settled_cash", "position_value", "fraction_value", "receivable_value"))
                self.assertEqual(total, Decimal(row["equity"]))
        self.assertEqual(report["status"], "software_fixture_complete")
        self.assertEqual(report["investment_trials"], 0)
        self.assertFalse(report["eligible_for_confirmation"])

    def test_rank_eleven_is_exited_and_ties_resolve_by_permanent_id(self):
        rows = [{"security_id": sid, "score": "0"} for sid in self.manifest["universe"]]
        chosen = select_targets(rows, {"SEC11"}, True, self.spec)
        self.assertNotIn("SEC11", chosen)
        data = self.data()
        for row in data.prices.values():
            row["momentum_close"] = "100"
        index = next(i for i, r in enumerate(data.calendar) if r["session"] == "2025-12-31")
        self.assertEqual([r["security_id"] for r in rank_stocks(data, index, self.spec)], self.manifest["universe"])

    def test_late_history_and_missing_candidate_block_instead_of_filtering(self):
        data = self.data()
        index = next(i for i, r in enumerate(data.calendar) if r["session"] == "2025-12-31")
        key = "SEC12", data.calendar[index - 200]["session"]
        data.prices[key]["available_at"] = "2026-01-01T00:00:00Z"
        with self.assertRaisesRegex(DataGap, "unavailable"):
            rank_stocks(data, index, self.spec)
        del data.prices[key]
        with self.assertRaisesRegex(DataGap, "missing price for SEC12"):
            rank_stocks(data, index, self.spec)

    def test_momentum_ignores_current_close_and_uses_exact_session_lags(self):
        data = self.data()
        index = next(i for i, r in enumerate(data.calendar) if r["session"] == "2025-12-31")
        before = rank_stocks(data, index, self.spec)
        data.prices["SEC01", "2025-12-31"]["momentum_close"] = "9999999999"
        self.assertEqual(before, rank_stocks(data, index, self.spec))

    def test_orders_are_frozen_before_open_and_adverse_gap_rejects_quantity(self):
        for row in self.inputs["prices"]:
            if row["session"] == "2026-01-02" and row["security_id"] == "SEC01":
                row.update(open="120", high="121", low="119", close="120")
        report = evaluate(self.data(), self.spec)
        arm = report["results"][0]
        first_order = arm["intentions"][0]["orders"][0]
        self.assertEqual(first_order["quantity"], 10)
        rejected = [r for r in arm["ledger"] if r["kind"] == "rejected_order"]
        self.assertEqual(rejected[0]["reason"], "opening_gap_exceeds_cash_or_cap")
        self.assertFalse(any(r["kind"] == "buy" and r["security_id"] == "SEC01" and r["session"] == "2026-01-02" for r in arm["ledger"]))

    def test_stop_includes_entry_bar_and_latches_at_close(self):
        for row in self.inputs["prices"]:
            if row["session"] == "2026-01-02":
                row.update(low="70", close="70")
        report = evaluate(self.data(), self.spec)
        arm = report["results"][0]
        stops = [r for r in arm["ledger"] if r["kind"] == "sell"]
        self.assertEqual(len(stops), 5)
        self.assertTrue(all(r["session"] == "2026-01-02" and r["price"] == "98" for r in stops))
        self.assertTrue(all(r["price_basis"] == "adverse_daily_range_proxy" for r in stops))
        self.assertTrue(all(r["price_basis"] == "vendor_daily_bar_open" for r in arm["ledger"] if r["kind"] == "buy"))
        self.assertEqual(arm["drawdown_latched_at"], "2026-01-02")
        self.assertEqual(sum(r["kind"] == "buy" for r in arm["ledger"]), 5)

    def test_event_cancels_previously_planned_suspended_stock_order(self):
        self.inputs["actions"] = [{"action_id": "cash", "kind": "cash_merger", "security_id": "SEC01",
                                    "session": "2026-01-02", "available_at": "2026-01-02T09:00:00-05:00",
                                    "amount": "100", "payable_date": None}]
        # The suspended security has no bar on the effective date.
        self.inputs["prices"] = [r for r in self.inputs["prices"] if not (r["security_id"] == "SEC01" and r["session"] >= "2026-01-02")]
        # Metadata describes the preannounced validity interval.
        self.inputs["securities"][0]["eligible_until"] = "2026-01-02"
        report = evaluate(self.data(), self.spec)
        rejected = [r for r in report["results"][0]["ledger"] if r["kind"] == "rejected_order"]
        self.assertEqual(rejected[0]["reason"], "corporate_action_requires_fresh_quantity")

    def test_bad_hash_unsafe_spec_and_nonfinite_price_generate_blocked_report(self):
        path = self.directory / "prices.json"
        path.write_text("[]")
        _, report = runner.run(self.manifest_path, SPEC_PATH, self.directory / "runs")
        self.assertEqual(report["status"], "blocked_market_data")
        self.assertIn("SHA-256 mismatch", report["failure_reasons"][0])
        self.assertTrue(all(v is None for v in report["metrics"].values()))
        self.spec["initial_cash"] = "100000"
        changed = self.directory / "spec.json"
        changed.write_text(json.dumps(self.spec))
        _, report = runner.run(self.manifest_path, changed, self.directory / "runs")
        self.assertIn("initial_cash", report["failure_reasons"][0])
        self.inputs["prices"][0]["close"] = "NaN"
        self.spec["initial_cash"] = "5000"
        with self.assertRaisesRegex(ValueError, "finite"):
            self.data()

    def test_unknown_or_late_action_and_short_settlement_calendar_block(self):
        self.inputs["actions"] = [{"action_id": "late", "kind": "dividend", "security_id": "SEC01",
                                    "session": "2026-01-02", "available_at": "2026-01-03T00:00:00Z",
                                    "amount": "1", "payable_date": "2026-02-01"}]
        with self.assertRaisesRegex(DataGap, "unavailable"):
            self.data()
        self.inputs["actions"] = []
        self.manifest["blocks"][0]["end"] = "2026-02-10"
        with self.assertRaisesRegex(DataGap, "settlement sessions"):
            self.data()

    def test_missing_bar_blocks_cash_only_window_before_either_arm_runs(self):
        self.manifest["blocks"][0].update(start="2026-01-06", end="2026-01-09")
        self.inputs["prices"] = [
            r for r in self.inputs["prices"]
            if (r["security_id"], r["session"]) != ("SEC12", "2026-01-07")
        ]
        write_inputs(self.directory, self.manifest, self.inputs)
        with patch.object(runner, "evaluate", side_effect=AssertionError("ran either arm")):
            _, report = runner.run(self.manifest_path, SPEC_PATH, self.directory / "runs")
        self.assertEqual(report["status"], "blocked_market_data")
        self.assertIn("missing price for SEC12 on 2026-01-07", report["failure_reasons"][0])

    def test_future_window_with_no_eligible_candidates_needs_completed_receipt(self):
        for security in self.inputs["securities"]:
            security["exchange"] = "XNAS"
        self.inputs["prices"] = []
        self.manifest["blocks"][0].update(start="2026-01-06", end="2026-01-09")
        write_inputs(self.directory, self.manifest, self.inputs)
        self.manifest["files"]["prices"]["received_at"] = "2026-01-06T22:00:00Z"
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(DataGap, "no completed price receipt through 2026-01-09"):
            load_dataset(self.manifest_path, self.spec)

    def test_check_inputs_validates_completed_files_without_scoring_strategies(self):
        with patch.object(runner, "evaluate", side_effect=AssertionError("ran either arm")):
            checked, report = runner.run(self.manifest_path, SPEC_PATH, self.directory / "runs", check_inputs=True)
        self.assertEqual(report["status"], "inputs_validated")
        self.assertEqual(report["results"], [])
        self.assertIsNone(report["metrics"]["net_return"])
        replayed, _ = runner.run(self.manifest_path, SPEC_PATH, self.directory / "runs")
        self.assertNotEqual(checked, replayed)

    def test_membership_end_between_signal_and_fill_prevents_entry(self):
        self.inputs["securities"][0]["eligible_until"] = "2026-01-02"
        self.inputs["prices"] = [
            r for r in self.inputs["prices"]
            if r["security_id"] != "SEC01" or r["session"] < "2026-01-02"
        ]
        arm = evaluate(self.data(), self.spec)["results"][0]
        self.assertTrue(any(r.get("reason") == "security_ineligible_at_open" for r in arm["ledger"]))
        self.assertFalse(any(r["kind"] == "buy" and r["security_id"] == "SEC01" for r in arm["ledger"]))

    def test_unknown_class_blocks_and_explicit_noncommon_is_reported(self):
        self.inputs["securities"][-1]["instrument_type"] = "UNKNOWN"
        with self.assertRaisesRegex(DataGap, "unresolved exchange or instrument"):
            evaluate(self.data(), self.spec)
        self.inputs["securities"][-1]["instrument_type"] = "PREFERRED_STOCK"
        report = evaluate(self.data(), self.spec)
        exclusions = report["results"][0]["decisions"][0]["eligibility"]
        self.assertEqual(exclusions[-1]["reason"], "outside_nyse_common_stock_mandate")
        self.inputs["securities"][-1]["eligible_until"] = "2020-01-01"
        with self.assertRaisesRegex(DataGap, "empty membership interval"):
            self.data()

    def test_warmup_atr_can_be_missing_and_does_not_use_future_values(self):
        self.inputs["prices"][0]["atr14"] = None
        report = evaluate(self.data(), self.spec)
        self.assertEqual(report["status"], "software_fixture_complete")
        for row in self.inputs["prices"]:
            if row["security_id"] == "SEC01" and row["session"] == "2025-12-31":
                row["atr14"] = None
        with self.assertRaisesRegex(ValueError, "finite decimal"):
            evaluate(self.data(), self.spec)

    def test_adapter_rejects_fixtures_and_maps_fraction_units_for_market_reports(self):
        from research.result_adapter import canonicalize_nyse

        report = evaluate(self.data(), self.spec)
        with self.assertRaisesRegex(ValueError, "market diagnostic"):
            canonicalize_nyse(report, arm="top_five", block="software")
        # This label change tests the adapter's format only. Inputs remain synthetic.
        report.update(status="historical_diagnostic_complete", data_kind="market")
        result = canonicalize_nyse(report, arm="top_five", block="software")
        self.assertEqual(result["drawdown"], report["results"][0]["drawdown"])
        self.assertEqual(result["metric_units"]["net_return"], "fraction")
        self.assertFalse(result["minimum_sample_met"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["executable_fills_verified"])
        self.assertIsNone(result["expected_net_profit_usd"])
        report["results"][0]["metric_units"]["drawdown"] = "percent"
        with self.assertRaisesRegex(ValueError, "unit mismatch"):
            canonicalize_nyse(report, arm="top_five", block="software")

    def test_corporate_actions_run_through_both_portfolios(self):
        actions = [
            {"action_id": "div", "kind": "dividend", "security_id": "SEC01", "session": "2026-01-05",
             "available_at": "2026-01-02T20:00:00Z", "amount": "1", "payable_date": "2026-01-12"},
            {"action_id": "split", "kind": "split", "security_id": "SEC02", "session": "2026-01-06",
             "available_at": "2026-01-02T20:00:00Z", "new_shares": "2", "old_shares": "1"},
            {"action_id": "cash", "kind": "cash_merger", "security_id": "SEC03", "session": "2026-01-07",
             "available_at": "2026-01-02T20:00:00Z", "amount": "100", "payable_date": None},
            {"action_id": "rename", "kind": "rename", "security_id": "SEC04", "session": "2026-01-08",
             "available_at": "2026-01-02T20:00:00Z", "old_ticker": "BEFORE", "new_ticker": "AFTER"}]
        self.inputs["actions"] = actions
        for row in self.inputs["prices"]:
            if row["security_id"] == "SEC02" and row["session"] >= "2026-01-06":
                row.update(open="50", close="50", high="50.5", low="49.5", atr14="0.5")
        report = evaluate(self.data(), self.spec)
        for arm in report["results"]:
            day = next(r for r in arm["equity_curve"] if r["session"] == "2026-01-06")
            self.assertEqual(day["positions"]["SEC02"]["quantity"], 20)
            self.assertEqual(day["receivables"]["div"]["amount"], "10")
            last = arm["equity_curve"][-1]
            self.assertIn("cash", last["receivables"])
            self.assertNotIn("SEC03", last["positions"])
            self.assertNotIn("div", last["receivables"])
            self.assertIn("cash_receipt_dates_unverified", arm["failure_reasons"])

    def test_actual_cli_succeeds_without_network_and_retries_preserve_artifact(self):
        output = self.directory / "runs"
        with patch("socket.socket.connect", side_effect=AssertionError("offline runner opened a socket")):
            destination, report = runner.run(self.manifest_path, SPEC_PATH, output)
        raw = (destination / "report.json").read_bytes()
        _, replayed = runner.run(self.manifest_path, SPEC_PATH, output)
        self.assertEqual(report, replayed)
        self.assertEqual(raw, (destination / "report.json").read_bytes())
        process = subprocess.run([sys.executable, str(ROOT / "scripts/run_nyse_offline_research.py"),
                                  "--manifest", str(self.manifest_path), "--output-dir", str(output)],
                                 cwd=self.directory, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        reply = json.loads(process.stdout)
        self.assertEqual(reply["run_id"], report["run_id"])
        self.assertFalse(reply["executable_fills_verified"])
        self.assertIsNone(reply["expected_net_profit_usd"])
        summary = (destination / "summary.md").read_text()
        self.assertIn("Execution is unverified", summary)
        self.assertIn("Simulated net return after stress", summary)
        self.assertIn("Simulated pre-tax P&L in USD", summary)
        _, inputs_only = runner.run(self.manifest_path, SPEC_PATH, output, check_inputs=True)
        self.assertEqual(inputs_only["status"], "inputs_validated")
        self.assertFalse(inputs_only["executable_fills_verified"])

    def test_cli_rejects_bad_input_with_exit_two_and_null_metrics(self):
        (self.directory / "prices.json").write_text("[]")
        output = self.directory / "blocked"
        process = subprocess.run([sys.executable, str(ROOT / "scripts/run_nyse_offline_research.py"),
                                  "--manifest", str(self.manifest_path), "--output-dir", str(output)],
                                 cwd=self.directory, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2, process.stderr)
        reply = json.loads(process.stdout)
        report = json.loads(Path(reply["report"]).read_text())
        self.assertEqual(report["status"], "blocked_market_data")
        self.assertTrue(all(value is None for value in report["metrics"].values()))
        self.assertFalse(report["results"])

    def test_concurrent_retries_publish_one_complete_report(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: runner.run(self.manifest_path, SPEC_PATH, self.directory / "runs"), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list((self.directory / "runs").iterdir())), 1)

    def test_future_or_incomplete_claims_do_not_change_evidence_status(self):
        self.manifest["evidence_role"] = "forward_confirmation"
        with self.assertRaisesRegex(DataGap, "development diagnostics"):
            self.data()
        self.manifest["evidence_role"] = "development_diagnostic"
        self.manifest["corporate_actions_complete_through"] = "2026-01-31"
        with self.assertRaisesRegex(DataGap, "corporate_actions_complete_through"):
            self.data()


if __name__ == "__main__":
    unittest.main()
