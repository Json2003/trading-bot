"""Small NYSE stock simulator for exercising the workstation, not finding alpha."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import RLock

from tradingbot_ibkr.execution.broker_base import Order
from tradingbot_ibkr.execution.paper_broker import PaperBroker

PRICES = {"IBM": 185.0, "JPM": 209.0, "KO": 65.0}
INITIAL_CASH = 5000.0
MAX_POSITION = 1000.0
COST_PER_SIDE = 0.0043
DEMO_START = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


class NyseWorkstationSimulator:
    """Generate demonstration bars and whole-share fills on the next bar."""

    def __init__(self, broker: PaperBroker) -> None:
        self._broker = broker
        self._bars: dict[str, deque[dict[str, object]]] = {s: deque(maxlen=108) for s in PRICES}
        self._cash = INITIAL_CASH
        self._peak = INITIAL_CASH
        self._step = 0
        self._lock = RLock()
        self.kill_switch_triggered = False
        self.kill_switch_reason = "NYSE demo drawdown reached 2%"
        for _ in range(78):
            self._append_bars()

    def _append_bars(self) -> None:
        self._step += 1
        day, minute = divmod(self._step - 1, 78)
        calendar_day = day + 2 * (day // 5)
        stamp = DEMO_START + timedelta(days=calendar_day, minutes=5 * minute)
        for index, (symbol, base) in enumerate(PRICES.items()):
            bars = self._bars[symbol]
            close = round(base * (1 + 0.011 * math.sin(self._step / 7 + index)
                                  + 0.005 * math.sin(self._step / 2.8 + 3 * index)
                                  + 0.00011 * (self._step % 120)), 2)
            opening = float(bars[-1]["close"]) if bars else base
            spread = base * (0.001 + 0.002 * abs(math.sin(self._step * 1.6 + index)))
            bars.append({
                "time": stamp.isoformat(), "open": opening, "close": close,
                "high": round(max(opening, close) + spread, 2),
                "low": round(min(opening, close) - spread * 0.8, 2),
                "volume": int(1200 + 9000 * abs(math.cos(self._step * 0.88 + index)) ** 2),
            })

    def _holdings(self) -> float:
        return sum(
            position.quantity * float(self._bars[position.symbol][-1]["close"])
            for position in self._broker.list_positions()
        )

    def step(self) -> None:
        with self._lock:
            self._append_bars()
            for order in self._broker.list_open_orders():
                opening = float(self._bars[order.symbol][-1]["open"])
                if order.side == "buy" and (
                    order.quantity * opening > MAX_POSITION
                    or order.quantity * opening * (1 + COST_PER_SIDE) > self._cash
                ):
                    self._broker.cancel_order(order.id)
                    continue
                fill = self._broker.fill_order(order.id, fill_price=opening)
                self._cash += fill.quantity * opening * (
                    1 - COST_PER_SIDE if order.side == "sell" else -(1 + COST_PER_SIDE)
                )
            equity = self._cash + self._holdings()
            self._peak = max(equity, self._peak)
            self.kill_switch_triggered = equity <= 0.98 * self._peak
            if self.kill_switch_triggered:
                return
            # A fixed cadence exercises simulated entries and exits. This is
            # intentionally independent of the paper lab and strategy scores.
            if self._step % 4 == 0:
                symbol = tuple(PRICES)[(self._step // 4) % len(PRICES)]
                position = next((p for p in self._broker.list_positions() if p.symbol == symbol), None)
                close = float(self._bars[symbol][-1]["close"])
                qty = position.quantity if position else math.floor(min(MAX_POSITION, self._cash) / (close * (1 + COST_PER_SIDE)))
                if qty > 0:
                    self._broker.submit_order(Order(
                        id=f"demo-{self._step}-{symbol}", symbol=symbol,
                        side="sell" if position else "buy", quantity=qty,
                    ))

    def workstation_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "source": "synthetic_nyse",
                "cash": self._cash,
                "equity": self._cash + self._holdings(),
                "starting_cash": INITIAL_CASH,
                "charts": {symbol: list(bars) for symbol, bars in self._bars.items()},
            }
