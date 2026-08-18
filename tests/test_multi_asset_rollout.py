import unittest

from scripts.evaluate_multi_asset_rollout import evaluate, load_policy


class MultiAssetRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()

    def test_milestone_ladder_and_growth_target(self):
        self.assertEqual(self.policy["growth_milestones_usd"],
                         [25000, 50000, 100000, 250000, 500000, 1000000])
        self.assertEqual(self.policy["target_monthly_growth_usd"], 1000)
        self.assertEqual(self.policy["target_months"], 24)

    def test_below_first_milestone_locks_all_sleeves(self):
        result = evaluate(24999.99, self.policy)
        self.assertFalse(result["milestone_reached"])
        self.assertEqual(result["milestone_usd"], 25000.0)
        self.assertFalse(result["live_trading_authorized"])
        self.assertTrue(all(sleeve["status"] == "locked_below_milestone"
                            for sleeve in result["sleeves"].values()))

    def test_milestone_opens_research_not_live_trading(self):
        result = evaluate(25000, self.policy)
        self.assertTrue(result["milestone_reached"])
        self.assertFalse(result["buffer_reached"])
        self.assertEqual(result["capital_growth_plan"]["next_milestone_usd"], 50000.0)
        self.assertFalse(result["live_trading_authorized"])

    def test_target_is_split_between_contribution_and_verified_pnl(self):
        result = evaluate(15000, self.policy, monthly_contribution=600)
        plan = result["capital_growth_plan"]
        self.assertEqual(plan["target_monthly_growth_usd"], 1000.0)
        self.assertEqual(plan["verified_net_pnl_required_usd"], 400.0)
        self.assertFalse(plan["return_assumptions_used"])

    def test_planner_tracks_contributions_without_return_forecasts(self):
        result = evaluate(15000, self.policy, monthly_contribution=500,
                          starting_equity=5000, net_contributions=10000,
                          verified_net_pnl=0)
        plan = result["capital_growth_plan"]
        self.assertEqual(plan["next_milestone_usd"], 25000.0)
        self.assertEqual(plan["gap_to_next_milestone_usd"], 10000.0)
        self.assertEqual(plan["contribution_only_months_to_next"], 20)
        self.assertEqual(plan["reconciliation"]["unreconciled_delta_usd"], 0.0)

    def test_policy_cannot_enable_live_or_leverage(self):
        broken = dict(self.policy)
        broken["live_trading_enabled"] = True
        with self.assertRaises(ValueError):
            from tempfile import NamedTemporaryFile
            import json
            with NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
                json.dump(broken, handle)
                handle.flush()
                from scripts.evaluate_multi_asset_rollout import load_policy
                from pathlib import Path
                load_policy(Path(handle.name))


if __name__ == "__main__":
    unittest.main()
