import unittest
from datetime import datetime, timedelta, timezone

from scripts.execution_model import STRESS_EXECUTION
from scripts.run_book_liquidation_confirmation import (
    BOOK_IMBALANCE_THRESHOLD,
    BOOK_PERSISTENCE_HOURS,
    book_side,
)


class BookLiquidationConfirmationTests(unittest.TestCase):
    def test_persistent_bid_pressure_is_long(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=index) for index in range(5)]
        book = {timestamp: 0.25 for timestamp in timestamps}
        self.assertEqual(book_side(book, timestamps, 2), 1)

    def test_persistent_ask_pressure_is_short(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=index) for index in range(5)]
        book = {timestamp: -0.25 for timestamp in timestamps}
        self.assertEqual(book_side(book, timestamps, 2), -1)

    def test_missing_hour_does_not_become_pressure(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=index) for index in range(5)]
        book = {timestamps[1]: 0.25, timestamps[2]: 0.25}
        self.assertEqual(book_side(book, timestamps, 2), 0)

    def test_parameters_and_costs_are_frozen(self) -> None:
        self.assertEqual(BOOK_IMBALANCE_THRESHOLD, 0.20)
        self.assertEqual(BOOK_PERSISTENCE_HOURS, 3)
        self.assertEqual(STRESS_EXECUTION.round_trip_bps, 86.0)


if __name__ == "__main__":
    unittest.main()
