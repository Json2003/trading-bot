import unittest
from datetime import datetime, timedelta, timezone

from scripts.execution_model import STRESS_EXECUTION
from scripts.run_liquidation_flow_reversal import (
    BASELINE_HOURS,
    DOMINANCE_THRESHOLD,
    EXTREME_MULTIPLIER,
    _signal,
    aggregate_hourly,
)


class LiquidationFlowReversalTests(unittest.TestCase):
    def test_hourly_aggregation_preserves_sides_and_notional(self) -> None:
        rows = [
            {
                "timestamp": datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc),
                "side": "SELL",
                "liquidation_usd": 100.0,
            },
            {
                "timestamp": datetime(2024, 1, 1, 0, 50, tzinfo=timezone.utc),
                "side": "BUY",
                "liquidation_usd": 25.0,
            },
        ]
        result = aggregate_hourly(rows)
        bucket = result[datetime(2024, 1, 1, 0, tzinfo=timezone.utc)]
        self.assertEqual(bucket["sell_usd"], 100.0)
        self.assertEqual(bucket["buy_usd"], 25.0)
        self.assertEqual(bucket["total_usd"], 125.0)

    def test_signal_is_causal_and_reverses_forced_flow(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=index) for index in range(BASELINE_HOURS + 2)]
        flow = {
            timestamp: {"buy_usd": 10.0, "sell_usd": 0.0, "total_usd": 10.0}
            for timestamp in timestamps[:BASELINE_HOURS]
        }
        signal_timestamp = timestamps[BASELINE_HOURS]
        flow[signal_timestamp] = {"buy_usd": 0.0, "sell_usd": 40.0, "total_usd": 40.0}
        self.assertEqual(_signal(flow, timestamps, BASELINE_HOURS), 1)
        self.assertEqual(_signal(flow, timestamps, BASELINE_HOURS + 1), 0)
        self.assertEqual(_signal(flow, timestamps, BASELINE_HOURS, {signal_timestamp.date()}), 0)
        self.assertEqual(EXTREME_MULTIPLIER, 3.0)
        self.assertEqual(DOMINANCE_THRESHOLD, 0.60)

    def test_shared_stress_execution_model_is_86_bps(self) -> None:
        self.assertEqual(STRESS_EXECUTION.round_trip_bps, 86.0)


if __name__ == "__main__":
    unittest.main()
