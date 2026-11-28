"""Verify the opening-price audit using local synthetic source examples."""
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from research.nyse_capture import digest, encode
from research.nyse_offline import DataGap
from research.nyse_open_audit import audit_open_prices, participant_opens


SESSION = "2026-11-20"
CTA_HEADER = (
    "Fri Nov 20 22:58:28 EDT 2026 prod\n"
    "Msg Cat/Type,TransTime,SeqNum,Symbol,instrumentType,partiId,prevClosePriceDate,openPrice\n"
)


class OpeningPriceAuditTests(unittest.TestCase):
    def test_uses_final_nyse_participant_open_and_preserves_missing(self):
        raw = (CTA_HEADER
               + "FractionalParticipantEODSummary,_20:00:20.102,_10,ABC,Equity,NYSE,_,101\n"
               + "FractionalParticipantEODSummary,_20:00:20.95,_11,ABC,Equity,NYSE,_,100\n"
               + "FractionalConsolidatedEODSummary,_20:00:21.,_12,ABC,Equity,NYSE,_,900\n"
               + "FractionalParticipantEODSummary,_20:00:21.,_13,ABC,Equity,ARCA,_,200\n"
               + "FractionalParticipantEODSummary,_20:00:19.,_14,XYZ,Equity,NYSE,_,150\n"
               + "FractionalParticipantEODSummary,_20:00:20.,_15,XYZ,Equity,NYSE,_11/19/2026,150\n"
               + "total 6\n").encode()
        plan = {"symbols": ["ABC", "XYZ"], "history_start": SESSION, "calendar_through": "2026-11-23",
                "closed_sessions": [], "early_closes": [], "decision_time_next_session_et": "09:00"}
        plan_hash = digest(encode(plan))
        review = {"schema_version": 1, "plan_sha256": plan_hash,
                  "reviewed_at": "2026-11-21T02:00:00Z", "membership_complete_through": SESSION,
                  "price_basis": "nominal_ohlc", "evidence": ["fixture"], "price_basis_evidence": ["fixture"],
                  "securities": [{"ticker": ticker, "security_id": ticker.lower(), "eligible_from": SESSION,
                                  "exchange": "XNYS", "instrument_type": "COMMON_STOCK",
                                  "available_at": "2026-11-20T14:00:00Z"} for ticker in plan["symbols"]]}
        bars = {(ticker, SESSION): {"open": "100", "available_at": "2026-11-21T01:00:00Z"}
                for ticker in plan["symbols"]}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            review_file, cta_file = base / "review.json", base / "cta.csv"
            review_file.write_bytes(encode(review))
            cta_file.write_bytes(raw)
            with patch("research.nyse_open_audit.captured_bars", return_value=(bars, [])):
                result = audit_open_prices(plan, plan_hash, base, review_file, cta_file, SESSION)
        self.assertEqual(result["status"], "incomplete_open_comparison")
        self.assertEqual(result["counts"], {"match": 0, "different": 1, "missing_open": 1})
        abc, xyz = result["comparisons"]
        self.assertEqual(abc["nyse_participant_open_usd"], "101")
        self.assertEqual(abc["cta_summary_time_label"], "_20:00:20.102")
        self.assertEqual(abc["venue_minus_daily_bps"], "100.00")
        self.assertIsNone(xyz["nyse_participant_open_usd"])
        self.assertEqual(result["cta_summary"]["sha256"], digest(raw))
        self.assertEqual(result["strategy_trials"], 0)
        self.assertFalse(result["executable_fills_verified"])

    def test_rejects_wrong_session_and_truncated_file(self):
        rows = "FractionalParticipantEODSummary,_20:00:20.,_10,ABC,Equity,NYSE,_,101\n"
        raw = (CTA_HEADER + rows + "total 2\n").encode()
        with self.assertRaisesRegex(DataGap, "is dated"):
            participant_opens(raw, date(2026, 11, 19), {"ABC"})
        with self.assertRaisesRegex(DataGap, "trailer does not match"):
            participant_opens(raw, date.fromisoformat(SESSION), {"ABC"})
        with self.assertRaisesRegex(DataGap, "missing its completeness trailer"):
            participant_opens((CTA_HEADER + rows).encode(), date.fromisoformat(SESSION), {"ABC"})
