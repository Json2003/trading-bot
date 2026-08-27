#!/usr/bin/env python3
"""Backtest and exact-reproduce the frozen 8-bps funding rule over one fixed year."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import run_funding_positioning_reversal as impl
except ModuleNotFoundError:
    import run_funding_positioning_reversal as impl


ONE_YEAR_START = datetime(2025, 8, 27, tzinfo=timezone.utc)
ONE_YEAR_END = datetime(2026, 8, 27, tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--btc-funding-path", type=Path, required=True)
    parser.add_argument("--eth-funding-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Use the existing frozen implementation; only the fixed evaluation window
    # changes. No parameter, execution-model, or signal-rule tuning is performed.
    impl.START = ONE_YEAR_START
    impl.END = ONE_YEAR_END
    aligned = impl.align_pair(
        impl.load_bars(args.btc_path),
        impl.load_bars(args.eth_path),
    )
    pair = [
        item for item in aligned
        if ONE_YEAR_START <= item.btc.timestamp < ONE_YEAR_END
    ]
    if not pair:
        raise ValueError("one-year window contains no completed aligned candles")
    if pair[-1].btc.timestamp < ONE_YEAR_END - impl.timedelta(hours=1):
        raise ValueError("one-year window is missing its final completed candle")

    funding = {
        "BTC": impl.load_funding(args.btc_funding_path),
        "ETH": impl.load_funding(args.eth_funding_path),
    }

    first_rows = impl.evaluate(pair, funding, 0, len(pair))
    repeat_rows = impl.evaluate(pair, funding, 0, len(pair))
    first = impl.summary(first_rows, 0, len(pair))
    repeat = impl.summary(repeat_rows, 0, len(pair))

    report = {
        "schema_version": 1,
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "hypothesis": "extreme funding indicates crowded positioning that reverses over the next eight hours",
        "frozen_parameters": {
            "assets": ["BTCUSDT", "ETHUSDT"],
            "funding_threshold_bps": impl.FUNDING_THRESHOLD * 10_000,
            "hold_hours": impl.HOLD_HOURS,
            "cooldown_hours": impl.COOLDOWN_HOURS,
            "direction": "positive funding short; negative funding long",
            "position_selection": "both assets independently; no leader selection",
            "notional": impl.NOTIONAL,
        },
        "execution_model": impl.STRESS_EXECUTION.as_dict(),
        "window": {
            "start": ONE_YEAR_START.isoformat(),
            "end_exclusive": ONE_YEAR_END.isoformat(),
            "last_completed_candle": pair[-1].btc.timestamp.isoformat(),
            "completed_candles_only": True,
            "non_overlapping_per_asset": True,
        },
        "funding_data": {
            "source": "Binance USD-M funding-rate archives plus completed current-month observations",
            "btc_rows": len(funding["BTC"]),
            "eth_rows": len(funding["ETH"]),
        },
        "one_year": first,
        "repeat": repeat,
        "trades": first_rows,
        "reproduction": {
            "identical_trade_rows": first_rows == repeat_rows,
            "identical_summary": first == repeat,
            "counts_as_new_evidence": False,
        },
        "status": "reproduced" if first_rows == repeat_rows and first == repeat else "mismatch",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "one_year_trades": first["trade_count"],
        "one_year_net_return_pct": first["net_return_pct"],
        "repeat_identical": report["reproduction"]["identical_trade_rows"]
        and report["reproduction"]["identical_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
