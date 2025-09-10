#!/usr/bin/env python3
import argparse, importlib, json, os, itertools, math, sys, inspect
from datetime import datetime

# Safe import guard: ensure repo root is on path and avoid shadow modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Avoid local files named like third-party libs from shadowing real packages
for shadow in ("pandas", "requests", "ccxt"):
    if shadow in sys.modules:
        del sys.modules[shadow]

# Force site-packages ahead of repo paths to import real pandas/requests
try:
    import importlib.util as _ilut
    _req_spec = _ilut.find_spec("requests")
    if _req_spec is not None and getattr(_req_spec, "origin", "").endswith("requests.py"):
        raise ImportError("Local requests.py shadowing detected")
except Exception:
    site_paths = [p for p in sys.path if "site-packages" in p]
    non_site = [p for p in sys.path if "site-packages" not in p]
    sys.path[:] = site_paths + non_site

import pandas as pd

from backtest.exchange import load_csv, fetch_ccxt
from backtest.engine import ExecConfig, run_backtest
from backtest.metrics import summarize

def load_strategy(spec: str):
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)

def filter_kwargs(fn, kwargs: dict) -> dict:
    """Keep only kwargs accepted by the strategy function signature."""
    try:
        sig = inspect.signature(fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs

def wrap_strategy(fn, kwargs):
    # returns a function(df) -> df_with_signals, injecting kwargs
    fkwargs = filter_kwargs(fn, kwargs or {})
    def _inner(df):
        return fn(df, **fkwargs)
    return _inner

def monthlyized_trades(num_trades: int, start: str, end: str) -> float:
    try:
        s = pd.to_datetime(start); e = pd.to_datetime(end)
        months = max(((e - s).days / 30.4375), 0.01)
        return num_trades / months
    except Exception:
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    # Data source
    ap.add_argument("--source", choices=["csv","ccxt"], required=True)
    ap.add_argument("--path")
    ap.add_argument("--exchange")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe")
    ap.add_argument("--since")
    ap.add_argument("--until")

    # Strategy
    ap.add_argument("--strategy", default="backtest.strategies.sma_filtered:generate_signals")
    ap.add_argument("--strategy_args", default="", help="Comma-separated key=value pairs to pass to strategy")
    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--slow", type=int, default=21)
    ap.add_argument("--trend_ma", type=int, default=200)
    ap.add_argument("--cooldown", type=int, default=3)

    # Exit mode and grids (comma-separated)
    ap.add_argument("--exit_mode", choices=["bps","atr"], default="bps", help="Which exit type to sweep")
    ap.add_argument("--grid_atr_pct", default="0.30,0.35,0.40,0.45,0.50,0.55", help="Used only if strategy accepts atr_pctile")
    ap.add_argument("--grid_tp_bps",  default="100,120,140")
    ap.add_argument("--grid_sl_bps",  default="30,40,50")
    ap.add_argument("--grid_tp_atr",  default="2.0,2.5,3.0")
    ap.add_argument("--grid_sl_atr",  default="1.0,1.25,1.5")
    ap.add_argument("--grid_max_bars", default="0,8,12", help="0 disables timeout")

    # Costs & exec
    ap.add_argument("--fees_bps", type=float, default=10.0)
    ap.add_argument("--slip_bps", type=float, default=5.0)
    ap.add_argument("--notional", type=float, default=1.0)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--risk_per_trade", type=float, default=0.005)
    ap.add_argument("--max_notional_frac", type=float, default=1.0)

    # Output
    ap.add_argument("--out_csv", default="artifacts/sweep_results.csv")
    args = ap.parse_args()

    # Load data
    if args.source == "csv":
        if not args.path: raise SystemExit("--path required for --source csv")
        df = load_csv(args.path)
    else:
        for k in ["exchange","symbol","timeframe","since","until"]:
            if not getattr(args, k): raise SystemExit(f"--{k} required for --source ccxt")
        df = fetch_ccxt(args.exchange, args.symbol, args.timeframe, args.since, args.until)

    os.makedirs("artifacts", exist_ok=True)

    # Load strategy fn
    strat_fn = load_strategy(args.strategy)

    # Parse strategy args
    extra_kwargs = {}
    if args.strategy_args:
        for pair in args.strategy_args.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise SystemExit(f"Invalid --strategy_args item: {pair}")
            k, v = pair.split("=", 1)
            k = k.strip(); v = v.strip()
            try:
                if "." in v:
                    extra_kwargs[k] = float(v)
                else:
                    extra_kwargs[k] = int(v)
            except ValueError:
                extra_kwargs[k] = v

    # Prepare grids
    atr_grid = [float(x.strip()) for x in args.grid_atr_pct.split(",") if x.strip()]
    tp_grid  = [float(x.strip()) for x in args.grid_tp_bps.split(",") if x.strip()]
    sl_grid  = [float(x.strip()) for x in args.grid_sl_bps.split(",") if x.strip()]
    tp_atr_grid = [float(x.strip()) for x in args.grid_tp_atr.split(",") if x.strip()]
    sl_atr_grid = [float(x.strip()) for x in args.grid_sl_atr.split(",") if x.strip()]
    max_bars_grid = [int(x.strip()) for x in args.grid_max_bars.split(",") if x.strip()]

    # Base (non-swept) strategy kwargs (filtered later against strategy signature)
    # Avoid duplicate keys when user passes the same keys via --strategy_args
    _base_fixed_keys = {"fast", "slow", "trend_ma", "cooldown"}
    _extra_filtered = {k: v for k, v in extra_kwargs.items() if k not in _base_fixed_keys}
    base_kwargs = dict(
        fast=args.fast,
        slow=args.slow,
        trend_ma=args.trend_ma,
        cooldown=args.cooldown,
        **_extra_filtered,
    )

    # Does the strategy accept 'atr_pctile'? If not, collapse this grid.
    accepts_atr_pctile = "atr_pctile" in getattr(inspect.signature(strat_fn), "parameters", {})
    atr_grid_eff = atr_grid if accepts_atr_pctile else [None]

    rows = []

    # Build parameter product based on exit mode
    if args.exit_mode == "bps":
        combos = list(itertools.product(atr_grid_eff, tp_grid, sl_grid, max_bars_grid))
        total = len(combos)
        for i, (atr_pct, tp_bps, sl_bps, max_bars) in enumerate(combos, start=1):
            # Strategy wrapper per combo
            kw = dict(base_kwargs)
            if atr_pct is not None:
                kw["atr_pctile"] = atr_pct
            fn = wrap_strategy(strat_fn, kw)

            cfg = ExecConfig(
                fees_bps=args.fees_bps,
                slip_bps=args.slip_bps,
                notional=args.notional,
                allow_short=args.allow_short,
                risk_per_trade=args.risk_per_trade,
                max_notional_frac=args.max_notional_frac,
                # exits
                tp_bps=float(tp_bps), sl_bps=float(sl_bps),
                tp_atr_mult=0.0, sl_atr_mult=0.0,
                atr_period=14,
                max_bars=int(max_bars),
            )
            trades, equity, bar_ret = run_backtest(df, fn, cfg)
            m = summarize(trades, equity, bar_ret)
            tpm = monthlyized_trades(m["num_trades"], m["start"], m["end"])
            rows.append({
                # Data
                "exchange": args.exchange or "csv",
                "symbol": args.symbol or "csv",
                "timeframe": args.timeframe or "csv",
                "start": m["start"], "end": m["end"],
                # Strategy
                "strategy": args.strategy,
                "fast": args.fast, "slow": args.slow, "trend_ma": args.trend_ma,
                "cooldown": args.cooldown,
                "atr_pctile": atr_pct,
                # Exec
                "exit_mode": "bps",
                "tp_bps": float(tp_bps), "sl_bps": float(sl_bps),
                "tp_atr_mult": None, "sl_atr_mult": None,
                "fees_bps": args.fees_bps, "slip_bps": args.slip_bps,
                "notional": args.notional, "max_bars": int(max_bars),
                "risk_per_trade": args.risk_per_trade, "max_notional_frac": args.max_notional_frac,
                # Metrics
                "total_return": m["total_return"],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m["sharpe"], "sortino": m["sortino"],
                "profit_factor": m["profit_factor"], "win_rate": m["win_rate"],
                "avg_trade": m["avg_trade"], "num_trades": m["num_trades"],
                "trades_per_month": tpm,
            })
            print(f"[{i}/{total}] mode=bps ATR%={(atr_pct if atr_pct is not None else float('nan')):.2f} "
                  f"TP={tp_bps} SL={sl_bps} MB={max_bars}  PF={m['profit_factor']:.2f} "
                  f"Sharpe={m['sharpe']:.2f} DD={m['max_drawdown']:.2%} Trades={m['num_trades']}")

    else:  # ATR exit mode
        combos = list(itertools.product(atr_grid_eff, tp_atr_grid, sl_atr_grid, max_bars_grid))
        total = len(combos)
        for i, (atr_pct, tp_atr, sl_atr, max_bars) in enumerate(combos, start=1):
            kw = dict(base_kwargs)
            if atr_pct is not None:
                kw["atr_pctile"] = atr_pct
            fn = wrap_strategy(strat_fn, kw)

            cfg = ExecConfig(
                fees_bps=args.fees_bps,
                slip_bps=args.slip_bps,
                notional=args.notional,
                allow_short=args.allow_short,
                risk_per_trade=args.risk_per_trade,
                max_notional_frac=args.max_notional_frac,
                # exits
                tp_bps=0.0, sl_bps=0.0,
                tp_atr_mult=float(tp_atr), sl_atr_mult=float(sl_atr),
                atr_period=14,
                max_bars=int(max_bars),
            )
            trades, equity, bar_ret = run_backtest(df, fn, cfg)
            m = summarize(trades, equity, bar_ret)
            tpm = monthlyized_trades(m["num_trades"], m["start"], m["end"])
            rows.append({
                # Data
                "exchange": args.exchange or "csv",
                "symbol": args.symbol or "csv",
                "timeframe": args.timeframe or "csv",
                "start": m["start"], "end": m["end"],
                # Strategy
                "strategy": args.strategy,
                "fast": args.fast, "slow": args.slow, "trend_ma": args.trend_ma,
                "cooldown": args.cooldown,
                "atr_pctile": atr_pct,
                # Exec
                "exit_mode": "atr",
                "tp_bps": None, "sl_bps": None,
                "tp_atr_mult": float(tp_atr), "sl_atr_mult": float(sl_atr),
                "fees_bps": args.fees_bps, "slip_bps": args.slip_bps,
                "notional": args.notional, "max_bars": int(max_bars),
                "risk_per_trade": args.risk_per_trade, "max_notional_frac": args.max_notional_frac,
                # Metrics
                "total_return": m["total_return"],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m["sharpe"], "sortino": m["sortino"],
                "profit_factor": m["profit_factor"], "win_rate": m["win_rate"],
                "avg_trade": m["avg_trade"], "num_trades": m["num_trades"],
                "trades_per_month": tpm,
            })
            print(f"[{i}/{total}] mode=atr ATR%={(atr_pct if atr_pct is not None else float('nan')):.2f} "
                  f"TPx={tp_atr} SLx={sl_atr} MB={max_bars}  PF={m['profit_factor']:.2f} "
                  f"Sharpe={m['sharpe']:.2f} DD={m['max_drawdown']:.2%} Trades={m['num_trades']}")

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nSaved {len(df_out)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
