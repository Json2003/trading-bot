#!/usr/bin/env python3
from __future__ import annotations

"""Run strategy backtests across predefined market regimes.

This script shells out to scripts/run_backtest.py using the modern engine
flags and aggregates per-regime metrics into a summary JSON.

Example:
  python scripts/run_regime_sweep.py \
    --symbol BTC/USDT --timeframe 4h \
    --strat_path backtest.strategies.sma_filtered:generate_signals \
    --strat_args fast=8,slow=21 \
    --exit_args tp_atr_mult=3.0,sl_atr_mult=1.5,max_bars=12 \
    --fees_bps 10 --slip_bps 5 \
    --outdir artifacts/regime_sweep
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict


REGIMES: Dict[str, Dict[str, str]] = {
    "bull": {"since": "2023-10-01", "until": "2024-03-01"},
    "bear": {"since": "2022-05-01", "until": "2022-12-31"},
    "chop": {"since": "2021-06-01", "until": "2021-10-01"},
}


def _parse_val(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def run_bt(symbol, timeframe, since, until, strat_path, strat_args, exit_args,
           fees_bps, slip_bps, out_prefix, allow_short):
    # Prefer project venv python if present, else current interpreter
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vpy = os.path.join(repo_root, ".venv", "bin", "python")
    py = vpy if os.path.exists(vpy) else sys.executable
    args = [
        py, "scripts/run_backtest.py",
        "--source", "ccxt", "--exchange", "kucoin",
        "--symbol", symbol, "--timeframe", timeframe,
        "--since", since, "--until", until,
        "--strategy", strat_path,
        "--strategy_args", ",".join(f"{k}={v}" for k, v in strat_args.items()),
        "--fees_bps", str(fees_bps), "--slip_bps", str(slip_bps),
        "--tp_bps", "0", "--sl_bps", "0",
        "--tp_atr_mult", str(exit_args.get("tp_atr_mult", 0)),
        "--sl_atr_mult", str(exit_args.get("sl_atr_mult", 0)),
        "--atr_period", "14",
        "--max_bars", str(int(exit_args.get("max_bars", 12))),
        "--out_prefix", out_prefix,
    ]
    if allow_short:
        args.append("--allow_short")
    subprocess.check_call(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--strat_path", required=True, help="module:function, e.g., backtest.strategies.sma_filtered:generate_signals")
    ap.add_argument("--strat_args", required=True, help="k=v,k=v")
    ap.add_argument("--exit_args", default="tp_atr_mult=3.0,sl_atr_mult=1.5,max_bars=12")
    ap.add_argument("--fees_bps", type=float, default=10.0)
    ap.add_argument("--slip_bps", type=float, default=5.0)
    ap.add_argument("--outdir", default="artifacts/regime_sweep")
    ap.add_argument("--allow_short", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sa = {}
    for kv in (args.strat_args or "").split(","):
        if kv and "=" in kv:
            k, v = kv.split("=", 1)
            sa[k.strip()] = _parse_val(v)
    ea = {}
    for kv in (args.exit_args or "").split(","):
        if kv and "=" in kv:
            k, v = kv.split("=", 1)
            ea[k.strip()] = float(v)

    report = {}
    for name, w in REGIMES.items():
        pref = os.path.join(args.outdir, f"{name}")
        run_bt(
            args.symbol,
            args.timeframe,
            w["since"],
            w["until"],
            args.strat_path,
            sa,
            ea,
            args.fees_bps,
            args.slip_bps,
            pref,
            args.allow_short,
        )
        with open(pref + "_metrics.json") as f:
            report[name] = json.load(f)

    path = os.path.join(args.outdir, "summary.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
