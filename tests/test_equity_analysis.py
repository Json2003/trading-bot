import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("equity_analysis", ROOT / "research/equity_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class EquityAnalysisTests(unittest.TestCase):
    def setUp(self):
        price = np.arange(250, dtype=float) / 10 + 100
        self.frame = pd.DataFrame({"session": pd.bdate_range("2025-01-01", periods=250).strftime("%Y-%m-%d"),
                                   "open": price, "high": price + 1, "low": price - 1,
                                   "close": price, "volume": np.repeat(100000, 250)})
        self.args = dict(symbol="IBM", as_of_session=self.frame.session.iloc[-1],
                         received_at="2026-11-14T17:00:00+00:00", calculated_at="2026-11-14T17:00:01+00:00", risk_latched=False)

    def test_missing_history_is_unknown_not_a_sell(self):
        result = analysis.stock_context(self.frame.iloc[:20], **self.args)
        self.assertEqual(result["action"], "WAIT_INSUFFICIENT_DATA")
        self.assertEqual(result["indicators"], {})

    def test_indicator_values_and_risk_latch_are_separate(self):
        self.args["risk_latched"] = True
        result = analysis.stock_context(self.frame, **self.args)
        self.assertEqual(result["action"], "PAUSED_DRAWDOWN_LATCH")
        self.assertAlmostEqual(result["indicators"]["atr_14"], 2.0)
        self.assertGreater(result["indicators"]["sma_20"], result["indicators"]["sma_200"])
        self.assertFalse(result["execution_authorized"])

    def test_future_prices_do_not_change_asof_indicators(self):
        self.args["as_of_session"] = self.frame.session.iloc[220]
        original = analysis.stock_context(self.frame, **self.args)
        changed = self.frame.copy()
        changed.loc[221:, ["open", "high", "low", "close"]] *= 100
        self.assertEqual(original["indicators"], analysis.stock_context(changed, **self.args)["indicators"])

    def test_catalog_has_primary_links_and_separate_opinions(self):
        catalog = json.loads((ROOT / "research/education/trading_curriculum_v1.json").read_text())
        analysis.validate_catalog(catalog)
        self.assertEqual(len(catalog["sources"]), 16)
        self.assertTrue(any(s["kind"] == "expert_opinion" for s in catalog["sources"]))
        selected = analysis.select_lessons(catalog, ["technical_signals"])
        self.assertTrue(selected)
        self.assertTrue(all(s["source_url"].startswith("https://") for s in selected))
        catalog["sources"].append(catalog["sources"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.validate_catalog(catalog)


if __name__ == "__main__":
    unittest.main()
