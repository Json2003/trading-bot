#!/usr/bin/env python3
import argparse, importlib, os, sys, json

# Safe import guard: fix sys.path and purge local shadow modules before any third-party imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Avoid local files named like third-party libs from shadowing real packages
for shadow in ("pandas", "requests", "ccxt"):
    if shadow in sys.modules:
        del sys.modules[shadow]

# Preload real 'requests' early to prevent accidental import of repo-local requests.py
try:
    import importlib.util as _ilut
    _req_spec = _ilut.find_spec("requests")
    if _req_spec is not None and getattr(_req_spec, "origin", "").endswith("requests.py"):
        raise ImportError("Local requests.py shadowing detected")
    import requests as _requests  # noqa: F401
except Exception:
    # Force site-packages 'requests' by manipulating sys.path search order
    site_paths = [p for p in sys.path if "site-packages" in p]
    non_site = [p for p in sys.path if "site-packages" not in p]
    sys.path[:] = site_paths + non_site
    import requests as _requests  # noqa: F401

from backtest.exchange import load_csv, fetch_ccxt
from backtest.engine import ExecConfig, run_backtest
from backtest.metrics import summarize


def dynamic_strategy(spec: str):
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csv","ccxt"], required=True)
    p.add_argument("--path")
    p.add_argument("--exchange")
    p.add_argument("--symbol")
    p.add_argument("--timeframe")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--strategy", default="backtest.strategies.sample_strategy:generate_signals")
    p.add_argument("--strategy_args", default="", help="Comma-separated key=value pairs passed to strategy, e.g. fast=5,slow=20")
    p.add_argument("--fees_bps", type=float, default=10.0)
    p.add_argument("--slip_bps", type=float, default=5.0)
    p.add_argument("--notional", type=float, default=1.0)
    # exits
    p.add_argument("--tp_bps", type=float, default=0.0)
    p.add_argument("--sl_bps", type=float, default=0.0)
    p.add_argument("--tp_atr_mult", type=float, default=0.0)
    p.add_argument("--sl_atr_mult", type=float, default=0.0)
    p.add_argument("--atr_period", type=int, default=14)
    # dynamic stops
    p.add_argument("--break_even_atr", type=float, default=0.0, help="Move stop to entry after +X*ATR in favor")
    p.add_argument("--trail_atr_mult", type=float, default=0.0, help="Trail stop by X*ATR (0 disables)")
    p.add_argument("--trail_method", choices=["atr","donchian"], default="atr")
    p.add_argument("--trail_ref", choices=["best","close"], default="best")
    p.add_argument("--donch_mid_n", type=int, default=0)
    # partial TP / payday
    p.add_argument("--tp_r_multiple", type=float, default=0.0)
    p.add_argument("--partial_tp_frac", type=float, default=0.0)
    p.add_argument("--lock_in_r_after_tp", type=float, default=0.0)
    # pullback/structure exit
    p.add_argument("--pullback_ema_len", type=int, default=0)
    p.add_argument("--pullback_atr_mult", type=float, default=0.0)
    p.add_argument("--pullback_confirm", type=int, default=0)
    # momentum/timebox
    p.add_argument("--min_rr_by_bars_r", type=float, default=0.0)
    p.add_argument("--min_rr_by_bars_n", type=int, default=0)
    # risk/sizing
    p.add_argument("--risk_per_trade", type=float, default=0.005)
    p.add_argument("--max_notional_frac", type=float, default=1.0)
    p.add_argument("--allow_short", action="store_true")
    p.add_argument("--max_bars", type=int, default=0, help="Exit after N bars in trade (0=disabled)")
    args = p.parse_args()

    if args.source=="csv":
        if not args.path:
            raise SystemExit("--path is required for --source csv")
        df = load_csv(args.path)
    else:
        for name in (args.exchange, args.symbol, args.timeframe, args.since, args.until):
            if not name:
                raise SystemExit("--exchange, --symbol, --timeframe, --since, --until are required for --source ccxt")
        df = fetch_ccxt(args.exchange,args.symbol,args.timeframe,args.since,args.until)

    fn = dynamic_strategy(args.strategy)
    # Parse strategy args into kwargs
    kwargs = {}
    if args.strategy_args:
        for pair in args.strategy_args.split(','):
            if not pair:
                continue
            if '=' not in pair:
                raise SystemExit(f"Invalid --strategy_args item: {pair}")
            k, v = pair.split('=', 1)
            k = k.strip(); v = v.strip()
            # try to cast to int/float where possible
            try:
                if '.' in v:
                    v_cast = float(v)
                else:
                    v_cast = int(v)
            except ValueError:
                v_cast = v
            kwargs[k] = v_cast
    cfg = ExecConfig(
        fees_bps=args.fees_bps,
        slip_bps=args.slip_bps,
        # exits
        tp_bps=args.tp_bps,
        sl_bps=args.sl_bps,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        atr_period=args.atr_period,
        break_even_atr_mult=args.break_even_atr,
        trail_atr_mult=args.trail_atr_mult,
        trail_method=args.trail_method,
        trail_ref=args.trail_ref,
        donch_mid_n=args.donch_mid_n,
        tp_r_multiple=args.tp_r_multiple,
        partial_tp_frac=args.partial_tp_frac,
        lock_in_r_after_tp=args.lock_in_r_after_tp,
        pullback_ema_len=args.pullback_ema_len,
        pullback_atr_mult=args.pullback_atr_mult,
        pullback_confirm=args.pullback_confirm,
        min_rr_by_bars_r=args.min_rr_by_bars_r,
        min_rr_by_bars_n=args.min_rr_by_bars_n,
        # risk
        notional=args.notional,
        risk_per_trade=args.risk_per_trade,
        max_notional_frac=args.max_notional_frac,
        # misc
        allow_short=bool(args.allow_short),
        max_bars=int(args.max_bars),
    )
    # Wrap strategy to inject kwargs if provided
    if kwargs:
        def _fn(df_):
            return fn(df_, **kwargs)
        strat = _fn
    else:
        strat = fn
    trades, equity, bar_ret = run_backtest(df, strat, cfg)

    os.makedirs("artifacts", exist_ok=True)
    trades.to_csv("artifacts/trades.csv", index=False)
    equity.to_csv("artifacts/equity_curve.csv", index=False)

    metrics = summarize(trades, equity, bar_ret)
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2))


if __name__=="__main__":
    main()
