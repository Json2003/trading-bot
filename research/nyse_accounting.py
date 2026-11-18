"""Decimal cash accounting for offline NYSE research."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Mapping

ZERO = Decimal("0")


def decimal(value, label="value") -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite decimal")
    return result


def positive(value, label="value") -> Decimal:
    result = decimal(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def whole(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


@dataclass
class Position:
    quantity: int
    basis: Decimal
    income: Decimal
    entry_session: str
    stop: Decimal


@dataclass
class Receivable:
    amount: Decimal
    due: str | None
    kind: str


@dataclass
class Fraction:
    security_id: str
    quantity: Decimal
    basis: Decimal
    income: Decimal
    entry_session: str


class CashLedger:
    """Purchases use settled cash. Corporate entitlements retain their identity."""

    def __init__(self, sessions: list[str], initial_cash="5000", position_cap="1000", cost_bps="43"):
        if sessions != sorted(set(sessions)) or not sessions:
            raise ValueError("calendar must contain unique ordered sessions")
        self.sessions = sessions
        self.cash = positive(initial_cash, "initial cash")
        self.cap = positive(position_cap, "position cap")
        self.cost_rate = decimal(cost_bps, "cost bps") / 10000
        if self.cost_rate < 0 or self.cost_rate >= 1:
            raise ValueError("cost rate is out of range")
        self.positions: dict[str, Position] = {}
        self.receivables: dict[str, Receivable] = {}
        self.fractions: dict[str, Fraction] = {}
        self.events: list[dict] = []
        self.realizations: list[dict] = []
        self.costs = ZERO
        self.traded_notional = ZERO
        self.applied_actions: dict[str, dict] = {}
        self.session: str | None = None

    def log(self, kind: str, **fields) -> None:
        self.events.append({"session": self.session, "kind": kind, **fields})

    def begin(self, session: str) -> None:
        if session not in self.sessions or (self.session and session <= self.session):
            raise ValueError("ledger sessions must move forward on the supplied calendar")
        self.session = session
        for claim_id, claim in list(self.receivables.items()):
            if claim.due is not None and claim.due <= session:
                self.cash += claim.amount
                del self.receivables[claim_id]
                self.log("cash_received", claim_id=claim_id, amount=str(claim.amount), source=claim.kind)

    def settlement_session(self, session: str) -> str:
        if session < "2017-09-05":
            raise ValueError("settlement rules before 2017-09-05 are unsupported")
        lag = 1 if session >= "2024-05-28" else 2
        index = self.sessions.index(session) + lag
        if index >= len(self.sessions):
            raise ValueError("calendar needs post-window settlement sessions")
        return self.sessions[index]

    def buy(self, security_id: str, quantity: int, price, stop) -> None:
        if self.session not in self.sessions:
            raise ValueError("begin a calendar session before buying")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("purchase quantity must be positive whole shares")
        if security_id in self.positions:
            raise ValueError("new entries cannot replace an existing position")
        price = positive(price, "purchase price")
        stop = positive(stop, "entry stop")
        notional = price * quantity
        fee = notional * self.cost_rate
        if stop >= price or notional > self.cap or notional + fee > self.cash:
            raise ValueError("purchase violates stop, position cap or settled cash")
        self.cash -= notional + fee
        self.costs += fee
        self.traded_notional += notional
        self.positions[security_id] = Position(quantity, notional + fee, ZERO, self.session, stop)
        self.log("buy", security_id=security_id, quantity=quantity, price=str(price), fee=str(fee))

    def sell(self, security_id: str, quantity: int, price, reason: str) -> None:
        pos = self.positions[security_id]
        if isinstance(quantity, bool) or not isinstance(quantity, int) or not 0 < quantity <= pos.quantity:
            raise ValueError("sale quantity must be held whole shares")
        price = positive(price, "sale price")
        due = self.settlement_session(self.session)
        fraction = Decimal(quantity) / pos.quantity
        basis, income = pos.basis * fraction, pos.income * fraction
        notional = price * quantity
        fee = notional * self.cost_rate
        claim_id = f"sale:{len(self.events)}:{security_id}"
        self.receivables[claim_id] = Receivable(notional - fee, due, "sale")
        pos.quantity -= quantity
        pos.basis -= basis
        pos.income -= income
        if not pos.quantity:
            del self.positions[security_id]
        self.costs += fee
        self.traded_notional += notional
        self.realize(security_id, notional - fee - basis + income, pos.entry_session, reason)
        self.log("sell", security_id=security_id, quantity=quantity, price=str(price), fee=str(fee),
                 settlement_session=due, claim_id=claim_id, reason=reason)

    def realize(self, security_id: str, pnl: Decimal, entry_session: str, reason: str) -> None:
        self.realizations.append({"security_id": security_id, "entry_session": entry_session,
                                  "exit_session": self.session, "net_pnl": str(pnl), "reason": reason})

    def claim(self, claim_id: str, amount: Decimal, due: str | None, kind: str) -> None:
        if claim_id in self.receivables or claim_id in self.fractions:
            raise ValueError("claim identity is already used")
        if due is not None:
            date.fromisoformat(due)
            if due < self.session:
                raise ValueError("payment date precedes entitlement")
        if amount < 0:
            raise ValueError("negative entitlements are unsupported")
        if due is not None and due <= self.session:
            self.cash += amount
        else:
            self.receivables[claim_id] = Receivable(amount, due, kind)

    def apply_action(self, action: dict) -> None:
        """Apply one validated event exactly once, rejecting conflicting retries."""
        action_id = action["action_id"]
        if action_id in self.applied_actions:
            if self.applied_actions[action_id] != action:
                raise ValueError("conflicting corporate-action identity")
            return
        kind = action["kind"]
        sid = action["security_id"]
        pos = self.positions.get(sid)
        due = action.get("payable_date")
        if kind == "fraction_payment":
            claim = self.fractions.get(action["fraction_id"])
            if claim is None and any(f.security_id == sid for f in self.fractions.values()):
                raise ValueError("fraction payment does not identify the outstanding entitlement")
            if claim is not None:
                if claim.security_id != sid:
                    raise ValueError("fraction payment refers to another security")
                proceeds = claim.quantity * positive(action["price"], "fraction payment price")
                self.claim(action_id, proceeds, due, kind)
                self.realize(sid, proceeds - claim.basis + claim.income, claim.entry_session, kind)
                del self.fractions[action["fraction_id"]]
        elif kind == "rename":
            # Permanent security ID and cost basis survive a symbol change.
            pass
        elif pos is not None:
            if kind == "dividend":
                amount = pos.quantity * decimal(action["amount"], "dividend amount")
                self.claim(action_id, amount, due, kind)
                pos.income += amount
            elif kind in {"split", "stock_merger"}:
                target = action.get("successor_id", sid)
                if target != sid and target in self.positions:
                    raise ValueError("merger into an existing holding needs lot consolidation")
                if any(f.security_id == sid for f in self.fractions.values()):
                    raise ValueError("overlapping fractional corporate actions need reconciliation")
                numerator = positive(action["new_shares"], "new shares")
                denominator = positive(action["old_shares"], "old shares")
                entitlement = pos.quantity * numerator / denominator
                integer = whole(entitlement)
                remainder = entitlement - integer
                if remainder:
                    part = remainder / entitlement
                    self.fractions[action_id] = Fraction(target, remainder, pos.basis * part,
                                                        pos.income * part, pos.entry_session)
                    pos.basis *= 1 - part
                    pos.income *= 1 - part
                del self.positions[sid]
                if integer:
                    pos.quantity = integer
                    pos.stop = pos.stop * denominator / numerator
                    self.positions[target] = pos
            elif kind == "cash_merger":
                proceeds = pos.quantity * positive(action["amount"], "cash merger consideration")
                self.claim(action_id, proceeds, due, kind)
                self.realize(sid, proceeds - pos.basis + pos.income, pos.entry_session, kind)
                del self.positions[sid]
            else:
                raise ValueError("unsupported corporate action")
        elif kind not in {"dividend", "split", "stock_merger", "cash_merger"}:
            raise ValueError("unsupported corporate action")
        self.applied_actions[action_id] = dict(action)
        self.log("corporate_action", action=dict(action), applied_to_holding=pos is not None)

    def snapshot(self, marks: Mapping[str, object]) -> dict:
        ids = set(self.positions) | {f.security_id for f in self.fractions.values()}
        values = {sid: positive(marks[sid], f"mark for {sid}") for sid in ids}
        position_value = sum((p.quantity * values[sid] for sid, p in self.positions.items()), ZERO)
        fraction_value = sum((f.quantity * values[f.security_id] for f in self.fractions.values()), ZERO)
        receivable_value = sum((r.amount for r in self.receivables.values()), ZERO)
        if self.cash < 0 or any(p.quantity <= 0 for p in self.positions.values()):
            raise ValueError("cash or position invariant failed")
        return {"session": self.session, "settled_cash": str(self.cash),
                "receivables": {k: {"amount": str(v.amount), "due": v.due, "kind": v.kind}
                                for k, v in self.receivables.items()},
                "fractional_interests": {k: {"security_id": v.security_id, "quantity": str(v.quantity)}
                                         for k, v in self.fractions.items()},
                "positions": {k: {"quantity": v.quantity, "basis": str(v.basis), "stop": str(v.stop)}
                              for k, v in self.positions.items()},
                "position_value": str(position_value), "fraction_value": str(fraction_value),
                "receivable_value": str(receivable_value),
                "equity": str(self.cash + position_value + fraction_value + receivable_value)}
