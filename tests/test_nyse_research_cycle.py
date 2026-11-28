"""Execution controls for the NYSE-only diagnostic (not performance tests)."""
import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("nyse_cycle", ROOT / "scripts/run_nyse_research_cycle.py")
cycle = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(cycle)


class NyseCycleTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / "research/experiments/nyse_stock_cycle_v1.json").read_text())
        self.spec["symbols"] = ["IBM"]
        self.spec["max_holding_sessions"] = 100
        self.spec["drawdown_latch_fraction"] = 0.02
        self.frame = pd.DataFrame({
            "session": pd.bdate_range("2026-10-01", periods=12).strftime("%Y-%m-%d"),
            "open": [100.0] * 12, "high": [101.0] * 12, "low": [99.0] * 12,
            "close": [100.0] * 12, "volume": [100000] * 12,
            "atr": [2.0] * 12, "signal": [1] * 12, "dividend": [0.0] * 12})
        self.block = {"name": "test", "start": self.frame.session.iloc[1], "end": self.frame.session.iloc[-1]}

    def test_costs_reduce_buy_and_sell_cash(self):
        self.assertAlmostEqual(cycle.fill_cash(100, 10, "buy", 43), -1004.3)
        self.assertAlmostEqual(cycle.fill_cash(100, 10, "sell", 43), 995.7)
        result = cycle.replay({"IBM": self.frame}, self.spec, self.block)
        self.assertLess(result["net_pnl"], 0)
        self.assertAlmostEqual(result["net_pnl"], -result["execution_costs"])

    def test_entry_uses_previous_completed_signal(self):
        self.frame["signal"] = 0
        self.frame.loc[2, "signal"] = 1
        result = cycle.replay({"IBM": self.frame}, self.spec, self.block)
        self.assertEqual(result["trades"][0]["entry_session"], self.frame.session.iloc[3])
        self.assertEqual(result["trades"][0]["exit_session"], self.frame.session.iloc[4])

    def test_gap_stop_uses_worse_open_and_does_not_reenter(self):
        self.frame.loc[2, ["open", "low", "close"]] = [90.0, 89.0, 90.0]
        result = cycle.replay({"IBM": self.frame}, self.spec, self.block)
        first = result["trades"][0]
        self.assertEqual(first["reason"], "gap_stop")
        self.assertEqual(first["exit_price"], 90.0)
        self.assertTrue(all(t["entry_session"] != self.frame.session.iloc[2] for t in result["trades"][1:]))

    def test_dividends_only_accrue_on_preexisting_holdings(self):
        self.frame.loc[1, "dividend"] = 7.0
        self.frame.loc[2, "dividend"] = 1.0
        result = cycle.replay({"IBM": self.frame}, self.spec, self.block)
        trade = result["trades"][0]
        self.assertEqual(trade["dividend_accrual"], trade["quantity"])
        self.assertAlmostEqual(result["net_pnl"], sum(t["net_pnl"] for t in result["trades"]))

    def test_drawdown_latches_and_liquidates_following_open(self):
        self.frame.loc[2, ["open", "high", "low", "close"]] = [100.0, 100.0, 70.0, 70.0]
        self.frame["atr"] = 100.0
        self.spec["risk_fraction"] = 1.0
        result = cycle.replay({"IBM": self.frame}, self.spec, self.block)
        self.assertEqual(result["drawdown_latched_at"], self.frame.session.iloc[2])
        self.assertEqual(result["trades"][0]["exit_session"], self.frame.session.iloc[3])
        self.assertEqual(result["trade_count"], 1)

    def test_nasdaq_and_non_equity_rejected(self):
        bad = {"meta": {"symbol": "IBM", "exchangeName": "NMS", "instrumentType": "EQUITY", "currency": "USD"}}
        with self.assertRaisesRegex(ValueError, "exchangeName"):
            cycle.parse_history(bad, "IBM", self.spec)
        bad["meta"].update(exchangeName="NYQ", instrumentType="CRYPTOCURRENCY")
        with self.assertRaisesRegex(ValueError, "instrumentType"):
            cycle.parse_history(bad, "IBM", self.spec)

    def test_split_history_is_rejected(self):
        split = {"meta": {"symbol": "IBM", "exchangeName": "NYQ", "instrumentType": "EQUITY", "currency": "USD"},
                 "events": {"splits": {"1": {"splitRatio": "2:1"}}}}
        with self.assertRaisesRegex(ValueError, "split-bearing"):
            cycle.parse_history(split, "IBM", self.spec)

    def test_overlapping_stock_trades_are_not_independent_samples(self):
        rows = [{"entry_session": "2026-01-01", "exit_session": "2026-01-10"},
                {"entry_session": "2026-01-02", "exit_session": "2026-01-11"},
                {"entry_session": "2026-01-12", "exit_session": "2026-01-20"}]
        self.assertEqual(cycle.independent_count(rows), 2)


if __name__ == "__main__":
    unittest.main()
