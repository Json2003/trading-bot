#!/usr/bin/env python3
"""Research-only multi-strategy crypto suite with trade-level metrics.

This is a pre-registered comparison, not an optimizer and not a trading
service. It uses only closed candles for signals, fills at the next open,
one long position at a time, no leverage, and explicit base/stress costs.
"""
from __future__ import annotations
import argparse, json, math, statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

try:
    from scripts.run_momentum_volatility_research import Bar, load_bars
    from scripts.run_momentum_volatility_v3 import PairBar, align_pair
except ModuleNotFoundError:
    from run_momentum_volatility_research import Bar, load_bars
    from run_momentum_volatility_v3 import PairBar, align_pair

HOURS_PER_YEAR = 24 * 365
THREE_YEARS = 3 * 365 * 24

@dataclass(frozen=True)
class Costs:
    fee_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float
    @property
    def slip_bps(self) -> float:
        return self.spread_bps / 2 + self.slippage_bps + self.impact_bps
    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.fee_bps + self.slip_bps)
BASE = Costs(10, 4, 5, 2)
STRESS = Costs(20, 10, 10, 8)

def mean(xs):
    return sum(xs) / len(xs) if xs else math.nan

def sma(xs, n):
    out = [math.nan] * len(xs)
    for i in range(n - 1, len(xs)):
        out[i] = mean(xs[i - n + 1:i + 1])
    return out

def stdev(xs, n):
    out = [math.nan] * len(xs)
    for i in range(n - 1, len(xs)):
        w = xs[i - n + 1:i + 1]
        m = mean(w)
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / max(1, len(w) - 1))
    return out

def ema(xs, n):
    out = [math.nan] * len(xs)
    if len(xs) < n:
        return out
    value = mean(xs[:n])
    out[n - 1] = value
    alpha = 2 / (n + 1)
    for i in range(n, len(xs)):
        value = alpha * xs[i] + (1 - alpha) * value
        out[i] = value
    return out

def prior_max(xs, n, i):
    if i < n + 1:
        return math.nan
    return max(xs[i - n:i])

def asset_features(bars: list[Bar]) -> dict[str, list[float]]:
    closes = [float(b.close) for b in bars]
    returns = [math.nan] + [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    return {
        "close": closes,
        "ema100": ema(closes, 100),
        "ema200": ema(closes, 200),
        "sma48": sma(closes, 48),
        "std48": stdev(closes, 48),
        "ret24": [math.nan if i < 24 else closes[i] / closes[i - 24] - 1 for i in range(len(closes))],
        "ret48": [math.nan if i < 48 else closes[i] / closes[i - 48] - 1 for i in range(len(closes))],
        "vol24": stdev([0 if not math.isfinite(x) else x for x in returns], 24),
        "prior_high48": [prior_max([float(b.high) for b in bars], 48, i) for i in range(len(bars))],
    }

def signal(name: str, i: int, f: Mapping[str, Mapping[str, list[float]]]) -> tuple[str | None, float]:
    btc, eth = f["BTC"], f["ETH"]
    if i < 220:
        return None, 0.0
    def valid(*xs): return all(math.isfinite(float(x)) for x in xs)
    if name in {"trend_defensive", "volatility_target"}:
        candidates = []
        for s, x in (("BTC", btc), ("ETH", eth)):
            if valid(x["ret24"][i], x["ema200"][i]) and x["ret24"][i] > 0 and x["close"][i] > x["ema200"][i]:
                candidates.append((x["ret24"][i], s, x))
        if not candidates:
            return None, 0.0
        _, s, x = max(candidates)
        size = 1.0
        if name == "volatility_target" and valid(x["vol24"][i]) and x["vol24"][i] > 0:
            size = min(1.0, 0.01 / x["vol24"][i])
        return s, max(0.25, size)
    if name == "relative_strength":
        if not valid(btc["ret48"][i], eth["ret48"][i]):
            return None, 0.0
        leader = "BTC" if btc["ret48"][i] > eth["ret48"][i] else "ETH"
        x = f[leader]
        if x["ret48"][i] - f["ETH" if leader == "BTC" else "BTC"]["ret48"][i] < 0.02 or x["close"][i] <= x["ema100"][i]:
            return None, 0.0
        return leader, 1.0
    if name == "mean_reversion":
        scores = []
        for s, x in (("BTC", btc), ("ETH", eth)):
            if valid(x["close"][i], x["sma48"][i], x["std48"][i]) and x["std48"][i] > 0:
                scores.append(((x["close"][i] - x["sma48"][i]) / x["std48"][i], s))
        if not scores:
            return None, 0.0
        z, s = min(scores)
        return (s, 0.75) if z < -1.5 else (None, 0.0)
    if name == "breakout":
        candidates = []
        for s, x in (("BTC", btc), ("ETH", eth)):
            if valid(x["close"][i], x["prior_high48"][i], x["ema100"][i]) and x["close"][i] > x["prior_high48"][i] and x["close"][i] > x["ema100"][i]:
                candidates.append((x["ret48"][i] if valid(x["ret48"][i]) else -math.inf, s))
        return (max(candidates)[1], 1.0) if candidates else (None, 0.0)
    raise ValueError(f"unknown strategy {name}")

def run_strategy(pair: list[PairBar], name: str, costs: Costs, initial_balance: float = 75000.0, order_notional: float = 6000.0) -> dict[str, object]:
    bars = {"BTC": [x.btc for x in pair], "ETH": [x.eth for x in pair]}
    f = {s: asset_features(v) for s, v in bars.items()}
    cash, qty, symbol = initial_balance, 0.0, None
    pending: tuple[str | None, float] = (None, 0.0)
    equity = []
    trade_pnl = []
    execution_cost = 0.0
    entries = 0
    for i in range(220, len(pair)):
        if symbol is not None and pending[0] != symbol:
            price = bars[symbol][i].open
            proceeds = qty * price * (1 - costs.slip_bps / 10000) * (1 - costs.fee_bps / 10000)
            cash += proceeds
            trade_pnl.append(cash - initial_balance if not trade_pnl else cash - initial_balance - sum(trade_pnl))
            execution_cost += qty * price * (costs.fee_bps + costs.slip_bps) / 10000
            qty, symbol = 0.0, None
        if symbol is None and pending[0] is not None:
            s, size = pending
            price = bars[s][i].open
            notional = min(cash, order_notional * size)
            fill = price * (1 + costs.slip_bps / 10000)
            qty = notional / fill * (1 - costs.fee_bps / 10000)
            cash -= notional
            execution_cost += notional * (costs.fee_bps + costs.slip_bps) / 10000
            symbol, entries = s, entries + 1
        mark = cash + (qty * bars[symbol][i].close if symbol is not None else 0.0)
        equity.append(mark)
        pending = signal(name, i, f)
    if symbol is not None:
        price = bars[symbol][-1].close
        cash += qty * price * (1 - costs.slip_bps / 10000) * (1 - costs.fee_bps / 10000)
        execution_cost += qty * price * (costs.fee_bps + costs.slip_bps) / 10000
        qty, symbol = 0.0, None
    ending = cash
    peaks, dd = initial_balance, 0.0
    returns = []
    for a, b in zip([initial_balance] + equity[:-1], equity):
        if a > 0:
            returns.append(b / a - 1)
            peaks = max(peaks, b)
            dd = max(dd, (peaks - b) / peaks)
    avg, sd = mean(returns), stdev(returns, len(returns))
    gains = sum(x for x in trade_pnl if x > 0)
    losses = abs(sum(x for x in trade_pnl if x < 0))
    return {
        "strategy": name, "initial_balance": initial_balance, "ending_balance": ending,
        "pnl_quote": ending - initial_balance, "return_pct": (ending / initial_balance - 1) * 100,
        "max_drawdown_pct": dd * 100, "entries": entries, "trades": entries * 2,
        "sharpe_annualized": (avg / sd * math.sqrt(HOURS_PER_YEAR)) if sd > 0 else None,
        "profit_factor": (gains / losses) if losses > 0 else None,
        "execution_cost_quote": execution_cost, "round_trip_bps": costs.round_trip_bps,
    }

def run_suite(btc_path: Path, eth_path: Path, output: Path) -> dict[str, object]:
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    if len(pair) < THREE_YEARS:
        raise ValueError("need at least three years of aligned hourly candles")
    pair = pair[-THREE_YEARS:]
    names = ["trend_defensive", "relative_strength", "mean_reversion", "breakout", "volatility_target"]
    report = {
        "schema_version": 1, "research_only": True, "orders_placed": False,
        "leverage_enabled": False, "automatic_promotion": False,
        "window": {"start": pair[0].timestamp.isoformat(), "end": pair[-1].timestamp.isoformat(), "bars": len(pair), "completed_candles_only": True},
        "strategies": {name: {"base": run_strategy(pair, name, BASE), "stress": run_strategy(pair, name, STRESS)} for name in names},
        "gates": {"requires_positive_stress_median": True, "requires_non_overlapping_walk_forward": True, "requires_trade_level_metrics": True, "promotion_allowed": False},
        "limitations": ["full-sample results are discovery evidence, not confirmation", "no shorting or leverage", "no news/funding/basis feed in this suite"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--btc-path", type=Path, required=True); p.add_argument("--eth-path", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    r = run_suite(p.parse_args().btc_path, p.parse_args().eth_path, p.parse_args().output)
    print(json.dumps({"strategies": list(r["strategies"]), "window": r["window"]}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
