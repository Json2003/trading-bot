from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.microstructure_feature import align_asof, load_measurements
from scripts.evaluate_microstructure_feature import evaluate


def at(hour):
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)


class MicrostructureFeatureTests(unittest.TestCase):
    def test_uses_availability_time_and_expires_stale_measurement(self):
        rows = [(at(0), at(2), 7.0), (at(1), at(4), 9.0)]
        self.assertEqual(
            align_asof([at(1), at(2), at(3), at(4), at(7)], rows, max_age=timedelta(hours=3)),
            [(0.0, 1.0), (7.0, 0.0), (7.0, 0.0), (9.0, 0.0), (0.0, 1.0)],
        )

    def test_rejects_untimestamped_or_revised_measurements(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            path.write_text("timestamp,available_at,value\n2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,1\n2026-01-01T00:00:00Z,2026-01-01T02:00:00Z,2\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_measurements(path)
            path.write_text("timestamp,available_at,value\n2026-01-01,2026-01-02,1\n")
            with self.assertRaisesRegex(ValueError, "UTC offset"):
                load_measurements(path)

    def test_candidate_evaluation_requires_real_coverage(self):
        with TemporaryDirectory() as directory:
            bars = Path(directory) / "bars.csv"
            bars.write_text("decision_ts,next_bar_executable_return,baseline\n" + "".join(
                f"{at(i).isoformat()},{0.01 if i % 2 else -0.01},{i % 7}\n" for i in range(120)
            ))
            measurements = Path(directory) / "measurements.csv"
            measurements.write_text("timestamp,available_at,value\n" + f"{at(0).isoformat()},{at(0).isoformat()},1\n")
            with self.assertRaisesRegex(ValueError, "coverage"):
                evaluate(bars, measurements, ["baseline"], 1)
            measurements.write_text("timestamp,available_at,value\n" + "".join(
                f"{at(i).isoformat()},{at(i).isoformat()},{i % 3}\n" for i in range(120)
            ))
            report = evaluate(bars, measurements, ["baseline"], 1)
            self.assertEqual(report["status"], "research_only_no_promotion")
            self.assertEqual(report["cost_bps_per_entry"], 86)
            self.assertEqual(report["results"]["candidate"]["holdout"]["rows"], 24)


if __name__ == "__main__":
    unittest.main()
