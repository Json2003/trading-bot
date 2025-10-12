"""Run a toy multi-strategy backtest using CCXT market data."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import ccxt  # type: ignore

from tradingbot_core.strategy import Bar, OrderIntent, Strategy
from tradingbot_core.strategies import (
    CrossExArb,
    DCAMartingale,
    GridConfig,
    GridStrategy,
    MomentumEMA,
)

TAKER_BPS = 8  # 0.08%
SLIP_BPS = 5  # 0.05%


def fee_price(side: str, px: float) -> float:
    """Apply taker fee plus a simple slippage model to a fill price."""

    adj = (TAKER_BPS + SLIP_BPS) / 1e4
    return px * (1 + adj if side == "buy" else 1 - adj)


def fetch_ohlcv(
    ex: ccxt.Exchange, symbol: str, timeframe: str = "1h", since_days: int = 180
) -> List[Bar]:
    """Download historical OHLCV candles from a CCXT exchange."""

    since = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000)
    out: List[Bar] = []
    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not chunk:
            break
        for ts, o, h, l, c, v in chunk:
            out.append(Bar(ts, float(o), float(h), float(l), float(c), float(v)))
        since = chunk[-1][0] + 1
        if len(chunk) < 1000:
            break
        time.sleep(ex.rateLimit / 1000.0)
    return out


def assign_strategy(intent: OrderIntent) -> str:
    """Return the strategy name based on the idempotency key prefix."""

    key = intent.idemp_key
    if key.startswith("arb-"):
        return "arbitrage"
    if key.startswith("grid-"):
        return "grid"
    if key.startswith("mom-"):
        return "momentum"
    return "dca"


def run_backtest() -> None:
    """Execute the multi-strategy toy backtest and persist the results."""

    binance = ccxt.binance({"enableRateLimit": True})

    symbols = ["BTC/USDT", "ETH/USDT"]
    series: Dict[str, List[Bar]] = {
        sym: fetch_ohlcv(binance, sym, "1h", since_days=120) for sym in symbols
    }

    arb_series: Dict[str, List[Bar]] = {
        "BINANCE:BTC/USDT": series["BTC/USDT"],
        "COINBASE:BTC/USDT": [
            Bar(b.ts, b.open, b.high, b.low, b.close * 1.001, b.volume) for b in series["BTC/USDT"]
        ],
    }

    grid = GridStrategy(
        GridConfig(symbol="BTC/USDT", lower=50_000, upper=70_000, levels=15, quantity=0.005, geometric=True)
    )
    momentum = MomentumEMA("ETH/USDT", 12, 26, 0.5, order_qty=0.5)
    dca = DCAMartingale("BTC/USDT", base_qty=0.002, step_pct=2.0, max_steps=4)
    arbitrage = CrossExArb("BTC/USDT", "BINANCE", "COINBASE", min_edge_bps=15, qty=0.01)
    strategies: List[tuple[str, Strategy]] = [
        ("grid", grid),
        ("momentum", momentum),
        ("dca", dca),
        ("arbitrage", arbitrage),
    ]

    equity0 = 100_000.0
    alloc = {"grid": 0.25, "momentum": 0.20, "dca": 0.30, "arbitrage": 0.25}
    strat_cash = {name: equity0 * weight for name, weight in alloc.items()}
    portfolio_curve = [equity0]

    pos_qty: Dict[str, float] = {name: 0.0 for name, _ in strategies}

    n = min(len(series["BTC/USDT"]), len(series["ETH/USDT"]))
    for i in range(n):
        bars = {
            "BTC/USDT": series["BTC/USDT"][i],
            "ETH/USDT": series["ETH/USDT"][i],
            "BINANCE:BTC/USDT": arb_series["BINANCE:BTC/USDT"][i],
            "COINBASE:BTC/USDT": arb_series["COINBASE:BTC/USDT"][i],
        }

        intents: List[OrderIntent] = []
        for _, strat in strategies:
            intents.extend(strat.on_bar(bars))

        for intent in intents:
            symbol = intent.symbol.split(":")[-1]
            bar = bars[symbol]
            price = bar.close if intent.type == "market" else intent.limit_price or bar.close
            fill_price = fee_price(intent.side, price)
            notional = intent.qty * fill_price
            name = assign_strategy(intent)

            if intent.side == "buy":
                if strat_cash[name] >= notional:
                    strat_cash[name] -= notional
                    pos_qty[name] += intent.qty
            else:
                dq = min(pos_qty[name], intent.qty)
                if dq > 0:
                    strat_cash[name] += dq * fill_price
                    pos_qty[name] -= dq

        mtm = 0.0
        for name, _ in strategies:
            sym = "ETH/USDT" if name == "momentum" else "BTC/USDT"
            mtm += pos_qty[name] * bars[sym].close
        port_equity = sum(strat_cash.values()) + mtm
        portfolio_curve.append(port_equity)

    result = {
        "start": series["BTC/USDT"][0].ts if n else None,
        "end": series["BTC/USDT"][n - 1].ts if n else None,
        "equity_curve": portfolio_curve,
        "notes": "Toy backtest (fees+slip applied; simplified fills). Replace with your full engine when ready.",
    }

    output_path = Path("backtest_portfolio_multi_strategy.json")
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    total_pnl = portfolio_curve[-1] / portfolio_curve[0] - 1.0
    print(
        f"Portfolio Total PnL: {total_pnl * 100:.2f}%  "
        f"(from {portfolio_curve[0]:.2f} to {portfolio_curve[-1]:.2f})"
    )


if __name__ == "__main__":  # pragma: no cover - manual entry point
    run_backtest()
