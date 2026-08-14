#!/usr/bin/env python3
"""Run the repository's independent research avenues side by side.

This is a paper/backtest diagnostic. It does not connect to a broker or promote
any result. Each strategy family is reported independently so a strong result
from one path cannot hide losses in another.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.optimization.execution_model import ExecutionCostModel
from backtest.strategies.regime_momentum import generate_signals as regime_momentum_signals
from tradingbot_ibkr.research_context import NewsEvent, gate_signal_series
from backtest.optimization.research_loop import (
    NightlyResearchLoop,
    _simulate_arbitrage,
    _simulate_dca,
    _grid_signals_factory,
    _momentum_signals_factory,
    create_non_overlapping_windows,
)


def _evaluation(
    name: str,
    summary: dict[str, Any],
    trades: int,
    costs: ExecutionCostModel,
) -> dict[str, Any]:
    gross_return = float(summary.get("total_return", 0.0))
    return {
        "strategy": name,
        "gross_return": gross_return,
        "net_return": costs.net_return(gross_return, trades),
        "test_sharpe": float(summary.get("sharpe", 0.0)),
        "test_drawdown": float(summary.get("max_drawdown", 0.0)),
        "profit_factor": (
            None
            if math.isinf(float(summary.get("profit_factor", 0.0)))
            else float(summary.get("profit_factor", 0.0))
        ),
        "trades": int(trades),
    }


def run_matrix(
    data: pd.DataFrame,
    *,
    window_size: int,
    test_fraction: float,
    costs: ExecutionCostModel | None = None,
    news_events: list[NewsEvent] | None = None,
    expected_move_bps: float = 40.0,
) -> dict[str, Any]:
    costs = costs or ExecutionCostModel()
    windows = create_non_overlapping_windows(
        data.sort_values("timestamp").reset_index(drop=True),
        window_size=window_size,
        test_fraction=test_fraction,
    )
    loop = NightlyResearchLoop(windows, Path("/tmp/research-matrix-registry.json"), trials_range=(1, 1))
    results: list[dict[str, Any]] = []
    def news_wrap(builder):
        if not news_events:
            return builder
        def wrapped(frame):
            raw = builder(frame)
            gated, blocked = gate_signal_series(
                raw["signals"],
                frame["timestamp"],
                news_events,
                expected_move_bps=expected_move_bps,
                expected_cost_bps=costs.round_trip_fraction * 10_000,
            )
            output = pd.DataFrame({"signals": gated})
            output.attrs["news_blocked"] = blocked
            return output
        return wrapped
    grid_configs = [(8, 0.03), (12, 0.05), (20, 0.08)]
    momentum_configs = [(5, 15), (8, 21), (13, 34)]
    regime_configs = [(8, 21, 100), (13, 34, 200), (13, 55, 200), (21, 55, 300)]
    dca_configs = [(0.02, 3), (0.04, 4), (0.06, 6)]
    arb_edges = [8.0, 15.0, 25.0]

    for window in windows:
        for levels, span in grid_configs:
            ev = loop._evaluate_signal_strategy(
                window.train,
                window.test,
                news_wrap(_grid_signals_factory(levels, span)),
            )
            results.append({"window": window.name, **_evaluation("grid", ev.test_summary, ev.test_trades, costs), "params": {"levels": levels, "range_pct": span}})
        for fast, slow in momentum_configs:
            ev = loop._evaluate_signal_strategy(
                window.train,
                window.test,
                news_wrap(_momentum_signals_factory(fast, slow)),
            )
            results.append({"window": window.name, **_evaluation("momentum", ev.test_summary, ev.test_trades, costs), "params": {"fast": fast, "slow": slow}})
        for fast, slow, regime in regime_configs:
            ev = loop._evaluate_signal_strategy(
                window.train,
                window.test,
                news_wrap(lambda frame, fast=fast, slow=slow, regime=regime: regime_momentum_signals(
                    frame, fast=fast, slow=slow, regime=regime
                )),
            )
            results.append(
                {
                    "window": window.name,
                    **_evaluation("regime_momentum", ev.test_summary, ev.test_trades, costs),
                    "params": {"fast": fast, "slow": slow, "regime": regime},
                }
            )
        for step, layers in dca_configs:
            summary, _, trades = _simulate_dca(window.test, step, layers)
            results.append({"window": window.name, **_evaluation("dca", summary, trades, costs), "params": {"step_pct": step, "max_layers": layers}})
        for edge in arb_edges:
            summary, _, trades = _simulate_arbitrage(window.test, edge)
            results.append({"window": window.name, **_evaluation("arbitrage", summary, trades, costs), "params": {"edge_bps": edge}})

    families: dict[str, dict[str, Any]] = {}
    for family in sorted({item["strategy"] for item in results}):
        entries = [item for item in results if item["strategy"] == family]
        families[family] = {
            "windows": len({item["window"] for item in entries}),
            "positive_tests": sum(item["net_return"] > 0 for item in entries),
            "total_tests": len(entries),
            "best_net_return": max(item["net_return"] for item in entries),
            "best": max(entries, key=lambda item: (item["net_return"], item["test_sharpe"])),
        }
    return {
        "cost_model": {
            "spread_bps": costs.spread_bps,
            "slippage_bps": costs.slippage_bps,
            "commission_bps": costs.commission_bps,
            "per_fill_fraction": costs.per_fill_fraction,
        },
        "families": families,
        "results": results,
        "news_mode": "enabled" if news_events else "price_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--window-size", type=int, default=480)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--spread-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=8.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--news-csv", type=Path, help="CSV with timestamp,sentiment,impact,category,source")
    parser.add_argument("--expected-move-bps", type=float, default=40.0)
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset)
    if "timestamp" not in frame.columns and "ts" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["ts"], utc=True)
    costs = ExecutionCostModel(
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
    )
    events: list[NewsEvent] = []
    if args.news_csv:
        news = pd.read_csv(args.news_csv)
        required = {"timestamp", "sentiment", "impact"}
        missing = required - set(news.columns)
        if missing:
            raise SystemExit(f"news CSV missing columns: {sorted(missing)}")
        for row in news.to_dict("records"):
            events.append(
                NewsEvent(
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    sentiment=float(row["sentiment"]),
                    impact=float(row["impact"]),
                    category=str(row.get("category", "unknown")),
                    source=str(row.get("source", "unknown")),
                )
            )
    print(json.dumps(run_matrix(frame, window_size=args.window_size, test_fraction=args.test_fraction, costs=costs, news_events=events, expected_move_bps=args.expected_move_bps), indent=2, default=str))


if __name__ == "__main__":
    main()
