"""Economic accounting regressions. Prices and receipts are software fixtures."""
import unittest
from decimal import Decimal

from research.nyse_accounting import CashLedger


class CashLedgerTests(unittest.TestCase):
    def test_sale_cash_waits_for_t2_and_t1_settlement(self):
        cases = [(["2022-01-13", "2022-01-14", "2022-01-18", "2022-01-19"], "2022-01-18"),
                 (["2024-05-24", "2024-05-28", "2024-05-29", "2024-05-30"], "2024-05-29"),
                 (["2024-05-28", "2024-05-29", "2024-05-30"], "2024-05-29")]
        for sessions, due in cases:
            with self.subTest(sessions=sessions):
                ledger = CashLedger(sessions, initial_cash="1004.30")
                ledger.begin(sessions[0])
                ledger.buy("A", 10, "100", "98")
                ledger.sell("A", 10, "100", "test")
                self.assertEqual(ledger.cash, 0)
                self.assertEqual(ledger.costs, Decimal("8.60"))
                self.assertEqual(next(iter(ledger.receivables.values())).due, due)
                with self.assertRaisesRegex(ValueError, "settled cash"):
                    ledger.buy("B", 1, "100", "98")
                for session in sessions[1:sessions.index(due)]:
                    ledger.begin(session)
                    self.assertEqual(ledger.cash, 0)
                ledger.begin(due)
                self.assertEqual(ledger.cash, Decimal("995.70"))

    def test_fees_reserved_and_fractional_or_boolean_quantities_rejected(self):
        ledger = CashLedger(["2026-01-05", "2026-01-06", "2026-01-07"], initial_cash="1000")
        ledger.begin("2026-01-05")
        for quantity in (10, 1.5, True):
            with self.assertRaises(ValueError):
                ledger.buy("A", quantity, "100", "98")
        ledger.buy("A", 9, "100", "98")
        self.assertEqual(ledger.cash, Decimal("96.13"))

    def test_dds_dividend_does_not_become_cash_on_ex_date(self):
        ledger = CashLedger(["2021-11-24", "2021-11-26", "2021-12-14", "2021-12-15"])
        ledger.begin("2021-11-24")
        ledger.buy("DDS", 2, "100", "90")
        cash = ledger.cash
        ledger.begin("2021-11-26")
        action = {"action_id": "DDS-special", "kind": "dividend", "security_id": "DDS", "amount": "15", "payable_date": "2021-12-15"}
        ledger.apply_action(action)
        ledger.apply_action(dict(action))
        self.assertEqual(ledger.cash, cash)
        self.assertEqual(ledger.receivables["DDS-special"].amount, 30)
        self.assertEqual(ledger.positions["DDS"].income, 30)
        ledger.begin("2021-12-14")
        self.assertEqual(ledger.cash, cash)
        ledger.begin("2021-12-15")
        self.assertEqual(ledger.cash, cash + 30)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ledger.apply_action({**action, "amount": "16"})

    def test_dividend_ignores_shares_bought_after_entitlement(self):
        ledger = CashLedger(["2026-01-05", "2026-01-06", "2026-01-07"])
        ledger.begin("2026-01-05")
        ledger.apply_action({"action_id": "div", "kind": "dividend", "security_id": "A", "amount": "15", "payable_date": "2026-01-07"})
        ledger.buy("A", 2, "100", "98")
        self.assertFalse(ledger.receivables)
        self.assertEqual(ledger.positions["A"].income, 0)

    def test_slg_split_preserves_equity_and_separates_cash_in_lieu(self):
        sessions = ["2022-01-21", "2022-01-24", "2022-01-25", "2022-01-26"]
        ledger = CashLedger(sessions)
        ledger.begin(sessions[0])
        ledger.buy("SLG", 103, "5", "4")
        prior = ledger.snapshot({"SLG": "5"})
        cash = ledger.cash
        ledger.begin(sessions[1])
        ledger.apply_action({"action_id": "SLG-split", "kind": "split", "security_id": "SLG", "new_shares": "1", "old_shares": "1.03060"})
        current = ledger.snapshot({"SLG": "5.153"})
        self.assertEqual(ledger.positions["SLG"].quantity, 99)
        self.assertAlmostEqual(Decimal(prior["equity"]), Decimal(current["equity"]), places=20)
        self.assertEqual(ledger.cash, cash)
        self.assertEqual(ledger.positions["SLG"].stop, Decimal("4.12240"))
        fraction = ledger.fractions["SLG-split"].quantity
        ledger.begin(sessions[2])
        ledger.apply_action({"action_id": "SLG-cil", "kind": "fraction_payment", "security_id": "SLG", "fraction_id": "SLG-split", "price": "5.20", "payable_date": sessions[3]})
        self.assertEqual(ledger.cash, cash)
        self.assertFalse(ledger.fractions)
        ledger.begin(sessions[3])
        self.assertEqual(ledger.cash, cash + fraction * Decimal("5.20"))

    def test_cit_successor_fraction_retains_cost_basis_until_payment(self):
        ledger = CashLedger(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"])
        ledger.begin("2022-01-03")
        ledger.buy("CIT", 10, "50", "48")
        cash = ledger.cash
        ledger.begin("2022-01-04")
        ledger.apply_action({"action_id": "CIT-merger", "kind": "stock_merger", "security_id": "CIT", "successor_id": "FCNCA", "new_shares": "0.06200", "old_shares": "1"})
        self.assertFalse(ledger.positions)
        fraction = ledger.fractions["CIT-merger"]
        self.assertEqual(fraction.security_id, "FCNCA")
        self.assertEqual(fraction.quantity, Decimal("0.62000"))
        self.assertEqual(fraction.basis, Decimal("502.1500"))
        self.assertEqual(ledger.cash, cash)
        ledger.begin("2022-01-05")
        ledger.apply_action({"action_id": "CIT-cil", "kind": "fraction_payment", "security_id": "FCNCA", "fraction_id": "CIT-merger", "price": "800", "payable_date": None})
        self.assertEqual(ledger.cash, cash)
        self.assertEqual(ledger.receivables["CIT-cil"].amount, 496)
        self.assertEqual(Decimal(ledger.realizations[-1]["net_pnl"]), Decimal("-6.1500"))

    def test_cspr_cash_merger_retains_undated_entitlement(self):
        ledger = CashLedger(["2022-01-24", "2022-01-25", "2022-01-26"])
        ledger.begin("2022-01-24")
        ledger.buy("CSPR", 10, "6.80", "6")
        cash = ledger.cash
        ledger.begin("2022-01-25")
        ledger.apply_action({"action_id": "CSPR-cash", "kind": "cash_merger", "security_id": "CSPR", "amount": "6.90", "payable_date": None})
        self.assertEqual(ledger.receivables["CSPR-cash"].amount, 69)
        self.assertFalse(ledger.positions)
        self.assertEqual(ledger.cash, cash)
        ledger.begin("2022-01-26")
        self.assertEqual(ledger.cash, cash)
        self.assertEqual(ledger.costs, Decimal("0.29240"))

    def test_cry_rename_keeps_permanent_identity_and_has_no_turnover(self):
        ledger = CashLedger(["2022-01-21", "2022-01-24", "2022-01-25"])
        ledger.begin("2022-01-21")
        ledger.buy("issuer-class-1", 10, "20", "18")
        position = ledger.positions["issuer-class-1"]
        costs = ledger.costs
        ledger.begin("2022-01-24")
        ledger.apply_action({"action_id": "rename", "kind": "rename", "security_id": "issuer-class-1", "old_ticker": "CRY", "new_ticker": "AORT"})
        self.assertIs(ledger.positions["issuer-class-1"], position)
        self.assertEqual(ledger.costs, costs)

    def test_partial_sale_allocates_income_and_basis_once(self):
        ledger = CashLedger(["2026-01-05", "2026-01-06", "2026-01-07"])
        ledger.begin("2026-01-05")
        ledger.buy("A", 2, "100", "98")
        ledger.apply_action({"action_id": "income", "kind": "dividend", "security_id": "A", "amount": "15", "payable_date": None})
        ledger.sell("A", 1, "100", "trim")
        self.assertEqual(ledger.positions["A"].income, 15)
        self.assertEqual(ledger.positions["A"].basis, Decimal("100.43"))
        self.assertEqual(Decimal(ledger.realizations[-1]["net_pnl"]), Decimal("14.14"))
        state = ledger.snapshot({"A": 100})
        self.assertEqual(Decimal(state["equity"]), Decimal("5028.71"))


if __name__ == "__main__":
    unittest.main()
