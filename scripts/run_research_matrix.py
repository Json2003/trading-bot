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
import statistics
from typing import Any

import pandas as pd

from backtest.optimization.execution_model import ExecutionCostModel
from backtest.strategies.regime_momentum import generate_signals as regime_momentum_signals
from tradingbot_ibkr.research_context import NewsEvent, gate_signal_series
MIN_TEST_TRADES_PER_WINDOW = 5
MIN_MATRIX_WINDOWS = 3
MIN_POSITIVE_WINDOW_FRACTION = 2.0 / 3.0
MAX_FAMILY_DRAWDOWN = 0.20


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
    gross_return = float(summary.get("total_return", math.nan))
    sharpe = float(summary.get("sharpe", math.nan))
    drawdown = abs(float(summary.get("max_drawdown", math.nan)))
    raw_profit_factor = float(summary.get("profit_factor", math.nan))
    profit_factor = raw_profit_factor if math.isfinite(raw_profit_factor) else None
    finite_metrics = all(
        math.isfinite(value)
        for value in (gross_return, sharpe, drawdown)
    )
    net_return = (
        costs.net_return(gross_return, trades)
        if finite_metrics
        else math.nan
    )
    return {
        "strategy": name,
        "gross_return": gross_return,
        "net_return": net_return,
        "cost_drag": gross_return - net_return if finite_metrics else math.nan,
        "execution_cost_bps": costs.round_trip_fraction * 10_000.0,
        "test_sharpe": sharpe,
        "test_drawdown": drawdown,
        # An infinite PF is retained as null: a no-loss sample is not treated
        # as robust evidence by the family gate.
        "profit_factor": profit_factor,
        "trades": int(trades),
        "finite_metrics": finite_metrics and math.isfinite(net_return),
    }


def run_matrix(
    data: pd.DataFrame,
    *,
    window_size: int,
    test_fraction: float,
    costs: ExecutionCostModel | None = None,
    news_events: list[NewsEvent] | None = None,
    expected_move_bps: float = 40.0,
    minimum_test_trades: int = MIN_TEST_TRADES_PER_WINDOW,
    minimum_windows: int = MIN_MATRIX_WINDOWS,
) -> dict[str, Any]:
    if window_size < 2:
        raise ValueError("window_size must be at least two rows")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if minimum_test_trades < 1 or minimum_windows < 1:
        raise ValueError("matrix research minimums must be positive")
    if not math.isfinite(expected_move_bps) or expected_move_bps < 0:
        raise ValueError("expected_move_bps must be finite and non-negative")
    required_columns = {"timestamp", "close"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["timestamp"].isna().any() or not frame["close"].map(math.isfinite).all() or (frame["close"] <= 0).any():
        raise ValueError("dataset contains invalid timestamps or close prices")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    costs = costs or ExecutionCostModel()
    windows = create_non_overlapping_windows(
        frame,
        window_size=window_size,
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
        family_windows = sorted({item["window"] for item in entries})
        eligible = [
            item
            for item in entries
            if item["finite_metrics"] and item["trades"] >= minimum_test_trades
        ]
        net_returns = [float(item["net_return"]) for item in eligible]
        sharpes = [float(item["test_sharpe"]) for item in eligible]
        profit_factors = [
            float(item["profit_factor"])
            for item in eligible
            if item["profit_factor"] is not None
            and math.isfinite(float(item["profit_factor"]))
        ]
        positive_tests = sum(value > 0 for value in net_returns)
        gate_reasons: list[str] = []
        if len(family_windows) < minimum_windows:
            gate_reasons.append(
                f"only {len(family_windows)} test windows; {minimum_windows} required"
            )
        if len(eligible) != len(entries):
            gate_reasons.append(
                f"{len(entries) - len(eligible)} window results fail finite/sample gates"
            )
        if not net_returns or statistics.median(net_returns) <= 0:
            gate_reasons.append("median net return is not positive")
        if (
            not net_returns
            or positive_tests / len(net_returns) < MIN_POSITIVE_WINDOW_FRACTION
        ):
            gate_reasons.append("fewer than two thirds of eligible windows are profitable")
        if not sharpes or statistics.median(sharpes) <= 0:
            gate_reasons.append("median test Sharpe is not positive")
        if not profit_factors or statistics.median(profit_factors) < 1.05:
            gate_reasons.append("median finite profit factor is below 1.05")
        if eligible and max(float(item["test_drawdown"]) for item in eligible) > MAX_FAMILY_DRAWDOWN:
            gate_reasons.append(f"test drawdown exceeds {MAX_FAMILY_DRAWDOWN:.0%}")
        families[family] = {
            "windows": len(family_windows),
            "positive_tests": positive_tests,
            "total_tests": len(entries),
            "best_net_return": max(
                (float(item["net_return"]) for item in entries if item["finite_metrics"]),
                default=None,
            ),
            # This is intentionally descriptive only. It is never used by the
            # family gate because selecting the best test slice is test leakage.
            "best_test_result_descriptive_only": max(
                entries,
                key=lambda item: (
                    float(item["net_return"])
                    if item["finite_metrics"]
                    else -math.inf,
                    float(item["test_sharpe"])
                    if math.isfinite(float(item["test_sharpe"]))
                    else -math.inf,
                ),
            ),
            "median_net_return": statistics.median(net_returns) if net_returns else None,
            "median_test_sharpe": statistics.median(sharpes) if sharpes else None,
            "median_test_drawdown": statistics.median(
                [float(item["test_drawdown"]) for item in eligible]
            ) if eligible else None,
            "median_profit_factor": statistics.median(profit_factors) if profit_factors else None,
            "eligible_windows": len(eligible),
            "research_gate": {
                "pass": not gate_reasons,
                "failure_reasons": gate_reasons,
                "minimum_test_trades_per_window": minimum_test_trades,
                "minimum_windows": minimum_windows,
                "minimum_positive_window_fraction": MIN_POSITIVE_WINDOW_FRACTION,
                "maximum_test_drawdown": MAX_FAMILY_DRAWDOWN,
                "best_test_result_is_not_evidence": True,
            },
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
        "research_gates": {
            "best_test_slice_is_descriptive_only": True,
            "minimum_test_trades_per_window": minimum_test_trades,
            "minimum_windows": minimum_windows,
            "execution_costs_applied": True,
        },
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
