#!/usr/bin/env python3
"""
Adaptive Superbot: regime-aware candidate generation and evaluation.

Runs a local grid around regime-specific params, evaluates in-sample (IS)
and out-of-sample (OOS), ranks by a composite score, and writes:
- <outdir>/leaderboard.json
- <outdir>/active_config.json
- For each candidate and window: prefixed metrics JSONs, trades, equity.
"""
from __future__ import annotations
import os, json, argparse
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure site-packages precede repo to avoid local shadowing
import sys
sys.path = [p for p in sys.path if p not in ('', REPO_ROOT)] + [REPO_ROOT]

# Preload real libs
for _name in ("pandas", "requests", "ccxt"):
    if _name in sys.modules:
        del sys.modules[_name]

import importlib

from backtest.exchange import fetch_ccxt
from backtest.engine import ExecConfig, run_backtest
from backtest.metrics import summarize
from backtest.adaptive.param_policy import params_for_regime
from backtest.adaptive.candidate_gen import around


def parse_date(s: str) -> str:
    # pass through YYYY-MM-DD (scripts/run_backtest also accepts epoch, but we fetch here)
    return s


def wrap_strategy(spec: str, kwargs: dict):
    mod_name, fn_name = spec.split(":")
    fn = getattr(importlib.import_module(mod_name), fn_name)
    if kwargs:
        def _inner(df, _fn=fn, _kw=kwargs):
            return _fn(df, **_kw)
        return _inner
    return fn


def eval_window(exchange: str, symbol: str, timeframe: str, since: str, until: str,
                strategy_spec: str, strat_kwargs: dict, exit_kwargs: dict,
                fees_bps: float, slip_bps: float, risk_per_trade: float, allow_short: bool,
                out_prefix: str) -> dict:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    df = fetch_ccxt(exchange, symbol, timeframe, parse_date(since), parse_date(until))
    strat = wrap_strategy(strategy_spec, strat_kwargs)
    cfg = ExecConfig(
        fees_bps=fees_bps,
        slip_bps=slip_bps,
        tp_bps=0.0,
        sl_bps=0.0,
        tp_atr_mult=float(exit_kwargs.get("tp_atr_mult", 0.0)),
        sl_atr_mult=float(exit_kwargs.get("sl_atr_mult", 0.0)),
        atr_period=14,
        notional=1.0,
        risk_per_trade=risk_per_trade,
        max_notional_frac=1.0,
        allow_short=bool(allow_short),
        max_bars=int(exit_kwargs.get("max_bars", 0)),
    )
    trades, equity, bar_ret = run_backtest(df, strat, cfg)
    m = summarize(trades, equity, bar_ret)
    # write artifacts by prefix
    trades.to_csv(f"{out_prefix}_trades.csv", index=False)
    equity.to_csv(f"{out_prefix}_equity.csv", index=False)
    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    return m


def score_combo(m_is: dict, m_oos: dict) -> tuple[float, bool]:
    score = (
        0.60 * max(0.0, m_is.get("sharpe", 0.0)) +
        0.25 * max(0.0, m_oos.get("sharpe", 0.0)) +
        0.10 * max(0.0, m_is.get("profit_factor", 0.0) - 1.0) +
        0.05 * max(0.0, m_oos.get("profit_factor", 0.0) - 1.0)
    )
    hard_pass = (
        m_is.get("max_drawdown", -1.0) >= -0.15 and
        m_oos.get("max_drawdown", -1.0) >= -0.20 and
        m_oos.get("profit_factor", 0.0) >= 1.05
    )
    return score, hard_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--since", default="2023-10-01")
    ap.add_argument("--until", default="2024-03-01")
    ap.add_argument("--fees_bps", type=float, default=10.0)
    ap.add_argument("--slip_bps", type=float, default=5.0)
    ap.add_argument("--risk_per_trade", type=float, default=0.005)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--oos_since", default="2022-05-01")
    ap.add_argument("--oos_until", default="2022-12-31")
    ap.add_argument("--regime_hint", default="bull")
    ap.add_argument("--outdir", default="artifacts/superbot")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    regime = args.regime_hint
    strategy_spec, strat_args, exit_args = params_for_regime(regime, base="trend_adx_atr")
    candidates = around(strat_args, exit_args)

    leaderboard = []
    for i, (sa, ea) in enumerate(candidates):
        pref = os.path.join(args.outdir, f"cand_{i}")
        m_is = eval_window(
            exchange="kucoin", symbol=args.symbol, timeframe=args.timeframe,
            since=args.since, until=args.until,
            strategy_spec=strategy_spec, strat_kwargs=sa, exit_kwargs=ea,
            fees_bps=args.fees_bps, slip_bps=args.slip_bps,
            risk_per_trade=args.risk_per_trade, allow_short=args.allow_short,
            out_prefix=pref+"_is",
        )
        m_oos = eval_window(
            exchange="kucoin", symbol=args.symbol, timeframe=args.timeframe,
            since=args.oos_since, until=args.oos_until,
            strategy_spec=strategy_spec, strat_kwargs=sa, exit_kwargs=ea,
            fees_bps=args.fees_bps * 2, slip_bps=args.slip_bps * 2,
            risk_per_trade=args.risk_per_trade, allow_short=args.allow_short,
            out_prefix=pref+"_oos",
        )
        score, hard_pass = score_combo(m_is, m_oos)
        leaderboard.append({
            "idx": i, "regime": regime, "score": score, "pass_": hard_pass,
            "is": m_is, "oos": m_oos, "strat_args": sa, "exit_args": ea, "strat_path": strategy_spec,
        })

    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    with open(os.path.join(args.outdir, "leaderboard.json"), "w") as f:
        json.dump(leaderboard, f, indent=2)

    active = next((row for row in leaderboard if row["pass_"]), leaderboard[0])
    with open(os.path.join(args.outdir, "active_config.json"), "w") as f:
        json.dump(dict(
            symbol=args.symbol, timeframe=args.timeframe, regime=regime,
            strat_path=active["strat_path"], strat_args=active["strat_args"],
            exit_args=active["exit_args"], eval=active,
        ), f, indent=2)

    print(json.dumps({
        "promoted_idx": active["idx"], "regime": regime, "score": active["score"],
        "is": active["is"], "oos": active["oos"]
    }, indent=2))


if __name__ == "__main__":
    main()
