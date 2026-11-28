"""Exercise receipt-based preparation using synthetic inputs with known answers."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.nyse_capture import calendar, digest, encode
from research.nyse_offline import DataGap, load_dataset
from research.nyse_preparation import enrich_prices, prepare

ROOT = Path(__file__).resolve().parents[1]


class MarketPreparationTests(unittest.TestCase):
    def test_split_and_dividend_preserve_total_return_and_split_rescales_atr(self):
        days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(17)]
        bars = {}
        for i, day in enumerate(days):
            close = Decimal(100 if i < 15 else 20 if i == 15 else 19)
            half_range = Decimal(5 if i < 15 else 1)
            bars["ABC", day] = {"open": str(close), "high": str(close + half_range),
                                "low": str(close - half_range), "close": str(close), "volume": "10",
                                "available_at": day + "T23:00:00Z"}
        equity = {"ticker": "ABC", "security_id": "abc", "available_at": "2025-12-31T23:00:00Z"}
        events = [
            {"security_id": "abc", "session": days[15], "kind": "split", "new_shares": "5",
             "old_shares": "1", "available_at": days[14] + "T20:00:00Z"},
            {"security_id": "abc", "session": days[16], "kind": "dividend", "amount": "1",
             "available_at": days[16] + "T23:30:00Z"},
        ]
        rows = enrich_prices(bars, [equity], [{"session": day} for day in days], events)
        self.assertEqual(Decimal(rows[14]["atr14"]), 10)
        self.assertEqual(Decimal(rows[15]["atr14"]), 2)
        self.assertEqual(Decimal(rows[15]["momentum_close"]), 100)
        self.assertEqual(Decimal(rows[16]["momentum_close"]), 100)
        self.assertEqual(datetime.fromisoformat(rows[16]["available_at"]).minute, 30)
        self.assertEqual(Decimal(rows[16]["close"]), 19)

    def test_full_preparation_produces_hash_checked_manifest_and_rejects_late_warmup(self):
        plan = {
            "history_start": "2024-12-02", "calendar_through": "2026-01-08", "closed_sessions": [],
            "early_closes": [], "decision_time_next_session_et": "09:00", "symbols": ["ABC"],
            "dataset_id": "synthetic-capture", "universe_selection": "One synthetic class",
            "block": {"name": "fixture", "start": "2026-01-01", "end": "2026-01-06"},
            "calendar_sources": ["synthetic"],
        }
        plan_hash = digest(encode(plan))
        review = {
            "schema_version": 1, "plan_sha256": plan_hash, "reviewed_at": "2026-01-07T01:00:00Z",
            "evidence": ["synthetic"], "price_basis_evidence": ["synthetic"], "price_basis": "nominal_ohlc",
            "membership_complete_through": "2026-01-06", "corporate_actions_complete_through": "2026-01-06",
            "securities": [{"ticker": "ABC", "security_id": "abc", "available_at": "2025-12-29T12:00:00Z",
                            "eligible_from": "2025-12-29", "eligible_until": None, "exchange": "XNYS",
                            "instrument_type": "COMMON_STOCK", "currency": "USD"}],
            "actions": [],
        }
        bars = {("ABC", r["session"]): {"open": "100", "high": "101", "low": "99", "close": "100",
                                      "volume": "1000000", "available_at": r["close_at"]}
                for r in calendar(plan) if r["session"] <= "2026-01-06"}
        spec = json.loads((ROOT / "research/experiments/nyse_momentum_retention_v1.json").read_text())
        with TemporaryDirectory() as directory:
            output = Path(directory)
            review_path = output / "review.json"
            review_path.write_bytes(encode(review))
            provenance = [{"file": "capture.json", "sha256": "a" * 64, "revision": "revision-1",
                           "from": "2024-12-02", "through": "2026-01-06",
                           "partitions": [{"url": "https://example.invalid/source", "received_at": "2026-01-07T00:00:00Z",
                                           "sha256": "b" * 64}]}]
            with patch("research.nyse_preparation.captured_bars", return_value=(bars, provenance)):
                result = prepare(plan, plan_hash, output / "receipts", review_path, output, spec)
                self.assertEqual(result["status"], "ready_for_offline_replay")
                self.assertEqual(result["strategy_trials"], 0)
                self.assertEqual(len(load_dataset(Path(result["manifest"]), spec).prices), len(bars))
                manifest = json.loads(Path(result["manifest"]).read_text())
                self.assertEqual(manifest["source_evidence"]["capture_plan_sha256"], plan_hash)
                self.assertEqual(manifest["source_evidence"]["capture_receipts"], provenance)
                manifest.pop("source_evidence")
                with self.assertRaisesRegex(DataGap, "hash-linked capture plan"):
                    load_dataset(Path(result["manifest"]), spec, manifest_raw=encode(manifest))
                bars["ABC", "2025-12-30"]["available_at"] = datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat()
                with self.assertRaisesRegex(DataGap, "was unavailable"):
                    prepare(plan, plan_hash, output / "receipts", review_path, output, spec)
