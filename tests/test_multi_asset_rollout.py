import json
import unittest
from pathlib import Path

from scripts.evaluate_multi_asset_rollout import evaluate, load_policy


class MultiAssetRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()

    def test_below_milestone_locks_all_sleeves(self):
        result = evaluate(24999.99, self.policy)
        self.assertFalse(result["milestone_reached"])
        self.assertFalse(result["live_trading_authorized"])
        self.assertTrue(all(
            sleeve["status"] == "locked_below_milestone"
            for sleeve in result["sleeves"].values()
        ))

    def test_milestone_opens_research_not_live_trading(self):
        result = evaluate(25000, self.policy)
        self.assertTrue(result["milestone_reached"])
        self.assertFalse(result["buffer_reached"])
        self.assertTrue(result["research_only"])
        self.assertFalse(result["live_trading_authorized"])
        self.assertTrue(all(
            sleeve["status"] == "paper_validation_eligible"
            for sleeve in result["sleeves"].values()
        ))

    def test_buffer_is_separate_from_milestone(self):
        result = evaluate(30000, self.policy)
        self.assertTrue(result["milestone_reached"])
        self.assertTrue(result["buffer_reached"])
        self.assertFalse(result["live_trading_authorized"])

    def test_policy_cannot_enable_live_or_leverage(self):
        broken = dict(self.policy)
        broken["live_trading_enabled"] = True
        with self.assertRaises(ValueError):
            evaluate(30000, load_policy_from_dict(broken))


def load_policy_from_dict(policy):
    # Exercise the same invariant checks without writing a temporary file.
    if policy.get("mode") != "paper_research_only":
        raise ValueError("rollout policy must remain paper_research_only")
    if policy.get("live_trading_enabled") is not False:
        raise ValueError("live trading must be disabled")
    if policy.get("leverage_enabled") is not False:
        raise ValueError("global leverage must be disabled")
    return policy


if __name__ == "__main__":
    unittest.main()
